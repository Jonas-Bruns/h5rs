use std::ffi::{c_char, c_uint, c_void, CStr};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use hdf5_metno_sys::h5::{herr_t, H5open, HADDR_UNDEF};
use hdf5_metno_sys::h5d::{
    H5D_fill_time_t::*, H5D_fill_value_t::*, H5D_layout_t::*, H5D_space_status_t::*, *,
};
use hdf5_metno_sys::h5e::{H5E_auto2_t, H5Eget_auto2, H5Eset_auto2, H5E_DEFAULT};
use hdf5_metno_sys::h5f::{H5Fclose, H5Fget_access_plist, H5Fopen, H5F_ACC_RDONLY};
use hdf5_metno_sys::h5i::hid_t;
use hdf5_metno_sys::h5p::*;
use hdf5_metno_sys::h5s::{
    H5Sclose, H5Sget_simple_extent_dims, H5Sget_simple_extent_ndims, H5S_ALL,
};
use hdf5_metno_sys::h5t::*;
use hdf5_metno_sys::h5vl::H5VLget_connector_name;
use hdf5_metno_sys::h5z::{H5Z_FILTER_DEFLATE, H5Z_FILTER_SHUFFLE};
use parking_lot::ReentrantMutex;

use crate::{ByteOrder, DType, DatasetMetadata, Error, Filter, NumericType};

// Reentrant so transient IDs can be dropped inside a locked metadata operation.
static HDF5: ReentrantMutex<()> = ReentrantMutex::new(());
const MAX_INDEX_ENTRIES: usize = 65_536;
const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;
static INDEX_BYTES: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct File(Id);
pub(crate) struct Dataset {
    id: Id,
    rank: usize,
    index: OnceLock<Option<ChunkIndex>>,
    pub storage: Storage,
}

#[derive(Clone, Copy)]
pub(crate) enum Storage {
    Chunked,
    Contiguous(Option<u64>),
    Compact,
}

#[derive(Clone, Copy)]
pub(crate) struct ChunkLocation {
    pub offset: u64,
    pub size: u64,
    pub filter_mask: u32,
}

type ChunkEntry = ([u64; 3], ChunkLocation);

#[derive(Default)]
pub(crate) struct ChunkIndex {
    entries: Vec<ChunkEntry>,
    reserved_bytes: usize,
}

impl ChunkIndex {
    pub(crate) fn location(&self, origin: [u64; 3]) -> Option<ChunkLocation> {
        let index = self
            .entries
            .binary_search_by_key(&origin, |entry| entry.0)
            .ok()?;
        Some(self.entries[index].1)
    }

    fn push(&mut self, entry: ChunkEntry, max_entries: usize) -> Result<(), Error> {
        if self.entries.len() == max_entries {
            return Err(Error::Limit {
                kind: "dataset chunk address index",
                bytes: (max_entries + 1) * size_of::<ChunkEntry>(),
                limit: max_entries * size_of::<ChunkEntry>(),
            });
        }
        if self.entries.len() == self.entries.capacity() {
            let capacity = self
                .entries
                .len()
                .saturating_mul(2)
                .max(64)
                .min(max_entries);
            let additional = capacity - self.entries.len();
            let bytes = additional * size_of::<ChunkEntry>();
            let mut used = INDEX_BYTES.load(Ordering::Relaxed);
            loop {
                let requested = used + bytes;
                if requested > MAX_INDEX_BYTES {
                    return Err(Error::Limit {
                        kind: "process chunk address indexes",
                        bytes: requested,
                        limit: MAX_INDEX_BYTES,
                    });
                }
                match INDEX_BYTES.compare_exchange_weak(
                    used,
                    requested,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(current) => used = current,
                }
            }
            if let Err(error) = self.entries.try_reserve_exact(additional) {
                INDEX_BYTES.fetch_sub(bytes, Ordering::Relaxed);
                return Err(error.into());
            }
            self.reserved_bytes += bytes;
        }
        self.entries.push(entry);
        Ok(())
    }
}

impl Drop for ChunkIndex {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.entries));
        INDEX_BYTES.fetch_sub(self.reserved_bytes, Ordering::Relaxed);
    }
}

impl File {
    pub(crate) fn open(path: &CStr) -> Result<Self, Error> {
        Self::open_with_driver(path, |access| {
            // SAFETY: The caller supplies a live file access property list.
            check(unsafe { H5Pset_fapl_sec2(access) }, "set sec2 driver")?;
            Ok(())
        })
    }

    /// The object must remain alive until this file and its datasets close.
    #[cfg(feature = "s3")]
    pub(crate) unsafe fn open_s3(object: &crate::s3::Object) -> Result<Self, Error> {
        Self::open_with_driver(c"h5rs-object", |access| {
            // SAFETY: A live FAPL is supplied and the caller keeps the borrowed
            // object alive through closure of all associated HDF5 handles.
            let status = unsafe {
                h5rs_set_range_driver(access, ptr::from_ref(object).cast(), object.len())
            };
            check(status, "set S3 range driver")?;
            Ok(())
        })
    }

    fn open_with_driver(
        path: &CStr,
        set_driver: impl FnOnce(hid_t) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        with_hdf5(|| {
            // SAFETY: HDF5 is initialized and locked, path is NUL-terminated, and
            // every successful ID is owned by its corresponding close function.
            unsafe {
                let access = Id::new(
                    H5Pcreate(*H5P_CLS_FILE_ACCESS),
                    H5Pclose,
                    "file access plist",
                )?;
                set_driver(access.raw)?;
                let driver = H5Pget_driver(access.raw);
                if driver < 0 {
                    return Err(Error::Hdf5("get expected file driver"));
                }
                let file = Id::new(
                    H5Fopen(path.as_ptr(), H5F_ACC_RDONLY, access.raw),
                    H5Fclose,
                    "open file",
                )?;
                let mut connector = [0_u8; 7];
                let size = H5VLget_connector_name(
                    file.raw,
                    connector.as_mut_ptr().cast(),
                    connector.len(),
                );
                if size != 6 || connector != *b"native\0" {
                    return Err(Error::Unsupported("non-native VOL connector"));
                }
                let actual = Id::new(
                    H5Fget_access_plist(file.raw),
                    H5Pclose,
                    "get file access plist",
                )?;
                if H5Pget_driver(actual.raw) != driver {
                    return Err(Error::Unsupported("unexpected file driver"));
                }
                Ok(Self(file))
            }
        })
    }

    pub(crate) fn dataset(&self, name: &CStr) -> Result<(Dataset, DatasetMetadata), Error> {
        with_hdf5(|| {
            // SAFETY: The file ID is live, name is NUL-terminated, and
            // the callback is a static C ABI function with no borrowed state.
            unsafe {
                let access = Id::new(
                    H5Pcreate(*H5P_CLS_DATASET_ACCESS),
                    H5Pclose,
                    "dataset access plist",
                )?;
                check(
                    H5Pset_elink_cb(access.raw, Some(reject_external_link), ptr::null_mut()),
                    "set external link callback",
                )?;
                let id = Id::new(
                    H5Dopen2(self.0.raw, name.as_ptr(), access.raw),
                    H5Dclose,
                    "open dataset",
                )?;
                let (metadata, storage) = dataset_metadata(id.raw)?;
                let rank = metadata.shape.len();
                Ok((
                    Dataset {
                        id,
                        rank,
                        index: OnceLock::new(),
                        storage,
                    },
                    metadata,
                ))
            }
        })
    }
}

#[cfg(feature = "s3")]
extern "C" {
    fn h5rs_set_range_driver(access: hid_t, object: *const c_void, size: u64) -> herr_t;
}

impl Dataset {
    pub(crate) fn read_compact(&self, output: &mut [u8]) -> Result<(), Error> {
        with_hdf5(|| {
            // SAFETY: The live compact dataset is read into a buffer sized for
            // its full shape and exact file datatype by the caller.
            unsafe {
                let datatype = Id::new(H5Dget_type(self.id.raw), H5Tclose, "get datatype")?;
                check(
                    H5Dread(
                        self.id.raw,
                        datatype.raw,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        output.as_mut_ptr().cast(),
                    ),
                    "read compact dataset",
                )?;
            }
            Ok(())
        })
    }

    pub(crate) fn location_index(&self) -> Option<&ChunkIndex> {
        // Concurrent first reads share a single initialization. Remember when
        // discovery is unavailable so later reads can fall back immediately.
        self.index
            .get_or_init(|| self.build_index(MAX_INDEX_ENTRIES).ok())
            .as_ref()
    }

    #[cfg(feature = "s3")]
    pub(crate) fn prepare_s3_index(&self) -> Result<(), Error> {
        // Called on a fresh dataset before sharing the handle with readers.
        if self.index.get().is_some() {
            return Ok(());
        }
        let index = self.build_index(MAX_INDEX_BYTES / size_of::<ChunkEntry>())?;
        let _ = self.index.set(Some(index));
        Ok(())
    }

    #[cfg(feature = "s3")]
    pub(crate) fn prepared_index(&self) -> Option<&ChunkIndex> {
        self.index.get().and_then(Option::as_ref)
    }

    fn build_index(&self, max_entries: usize) -> Result<ChunkIndex, Error> {
        let mut build = IndexBuild {
            rank: self.rank,
            max_entries,
            error: None,
            index: ChunkIndex::default(),
        };
        with_hdf5(|| {
            // SAFETY: The dataset ID is live. The callback borrows build only
            // for the duration of this synchronous iteration.
            check(
                unsafe {
                    H5Dchunk_iter(
                        self.id.raw,
                        H5P_DEFAULT,
                        Some(record_location),
                        ptr::from_mut(&mut build).cast(),
                    )
                },
                "iterate chunk locations",
            )
        })?;
        if let Some(error) = build.error {
            return Err(error);
        }
        let mut index = build.index;
        index.entries.sort_unstable_by_key(|entry| entry.0);
        if index.entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::InvalidChunk("duplicate indexed chunk origins"));
        }
        Ok(index)
    }

    pub(crate) fn chunk_location(&self, origin: &[u64]) -> Result<Option<ChunkLocation>, Error> {
        if origin.len() != self.rank {
            return Err(Error::SelectionRank {
                expected: self.rank,
                actual: origin.len(),
            });
        }
        with_hdf5(|| {
            let mut offset = HADDR_UNDEF;
            let mut size = 0;
            let mut filter_mask = 0;
            // SAFETY: The caller supplies one coordinate per axis of this live
            // dataset. All three output pointers refer to correctly sized locals.
            check(
                unsafe {
                    H5Dget_chunk_info_by_coord(
                        self.id.raw,
                        origin.as_ptr(),
                        &mut filter_mask,
                        &mut offset,
                        &mut size,
                    )
                },
                "get chunk information",
            )?;
            match (offset, size) {
                (HADDR_UNDEF, 0) => Ok(None),
                (HADDR_UNDEF, _) | (_, 0) => {
                    Err(Error::InvalidChunk("inconsistent address or size"))
                }
                _ => Ok(Some(ChunkLocation {
                    offset,
                    size,
                    filter_mask,
                })),
            }
        })
    }
}

struct IndexBuild {
    rank: usize,
    max_entries: usize,
    error: Option<Error>,
    index: ChunkIndex,
}

unsafe extern "C" fn record_location(
    coords: *const u64,
    filter_mask: c_uint,
    offset: u64,
    size: u64,
    data: *mut c_void,
) -> std::ffi::c_int {
    // SAFETY: HDF5 gives us at least rank coordinates and passes through
    // the live IndexBuild pointer from the synchronous H5Dchunk_iter call.
    let build = unsafe { &mut *data.cast::<IndexBuild>() };
    if offset == HADDR_UNDEF && size == 0 {
        return 0;
    }
    let mut origin = [0; 3];
    origin[3 - build.rank..]
        .copy_from_slice(unsafe { std::slice::from_raw_parts(coords, build.rank) });
    let location = ChunkLocation {
        offset,
        size,
        filter_mask,
    };
    let result = if offset == HADDR_UNDEF || size == 0 {
        Err(Error::InvalidChunk("inconsistent indexed address or size"))
    } else {
        build.index.push((origin, location), build.max_entries)
    };
    if let Err(error) = result {
        build.error = Some(error);
        1
    } else {
        0
    }
}

extern "C" fn reject_external_link(
    _: *const c_char,
    _: *const c_char,
    _: *const c_char,
    _: *const c_char,
    _: *mut c_uint,
    _: hid_t,
    _: *mut c_void,
) -> herr_t {
    -1
}

// The caller holds HDF5 and owns a live dataset ID.
unsafe fn dataset_metadata(dataset: hid_t) -> Result<(DatasetMetadata, Storage), Error> {
    // SAFETY: Each obtained ID is checked before use. The rank is validated
    // before either dimension query, and fill is read only for 1–8 byte types.
    unsafe {
        let space = Id::new(H5Dget_space(dataset), H5Sclose, "get dataspace")?;
        let rank = H5Sget_simple_extent_ndims(space.raw);
        check(rank, "get dataset rank")?;
        if !matches!(rank, 1..=3) {
            return Err(Error::Unsupported("dataset rank outside 1 through 3"));
        }
        let mut shape = vec![0; rank as usize];
        check(
            H5Sget_simple_extent_dims(space.raw, shape.as_mut_ptr(), ptr::null_mut()),
            "get dataset dimensions",
        )?;

        let properties = Id::new(
            H5Dget_create_plist(dataset),
            H5Pclose,
            "get dataset creation plist",
        )?;
        let layout = H5Pget_layout(properties.raw);
        if layout == H5D_LAYOUT_ERROR {
            return Err(Error::Hdf5("get dataset layout"));
        }
        let mut chunks = Vec::new();
        let storage = match layout {
            H5D_CHUNKED => {
                chunks.resize(rank as usize, 0);
                if H5Pget_chunk(properties.raw, rank, chunks.as_mut_ptr()) != rank
                    || chunks.contains(&0)
                {
                    return Err(Error::Hdf5("get chunk dimensions"));
                }
                let mut options = 0;
                check(
                    H5Pget_chunk_opts(properties.raw, &mut options),
                    "get chunk options",
                )?;
                if options != 0 {
                    return Err(Error::Unsupported("non-default chunk options"));
                }
                Storage::Chunked
            }
            H5D_CONTIGUOUS => {
                if check(
                    H5Pget_external_count(properties.raw),
                    "get external storage count",
                )? != 0
                {
                    return Err(Error::Unsupported("external dataset storage"));
                }
                let mut status = H5D_SPACE_STATUS_ERROR;
                check(
                    H5Dget_space_status(dataset, &mut status),
                    "get allocation status",
                )?;
                if status == H5D_SPACE_STATUS_NOT_ALLOCATED {
                    Storage::Contiguous(None)
                } else {
                    let offset = H5Dget_offset(dataset);
                    if offset == HADDR_UNDEF {
                        return Err(Error::Hdf5("get contiguous offset"));
                    }
                    Storage::Contiguous(Some(offset))
                }
            }
            H5D_COMPACT => Storage::Compact,
            _ => return Err(Error::Unsupported("dataset layout")),
        };

        let datatype = Id::new(H5Dget_type(dataset), H5Tclose, "get datatype")?;
        let dtype = numeric_type(datatype.raw)?;
        let filters = filter_pipeline(properties.raw, dtype.size())?;
        let mut fill_status = H5D_FILL_VALUE_ERROR;
        let mut fill_time = H5D_FILL_TIME_ERROR;
        check(
            H5Pfill_value_defined(properties.raw, &mut fill_status),
            "get fill status",
        )?;
        check(
            H5Pget_fill_time(properties.raw, &mut fill_time),
            "get fill timing",
        )?;
        if !matches!(
            fill_status,
            H5D_FILL_VALUE_DEFAULT | H5D_FILL_VALUE_USER_DEFINED
        ) || !matches!(fill_time, H5D_FILL_TIME_ALLOC | H5D_FILL_TIME_IFSET)
        {
            return Err(Error::Unsupported("undefined or disabled fill values"));
        }
        let mut fill_value = vec![0; dtype.size()];
        check(
            H5Pget_fill_value(properties.raw, datatype.raw, fill_value.as_mut_ptr().cast()),
            "get fill value",
        )?;
        Ok((
            DatasetMetadata {
                shape,
                chunks,
                dtype,
                filters,
                fill_value,
            },
            storage,
        ))
    }
}

// The caller holds HDF5 and owns a live datatype ID.
unsafe fn numeric_type(datatype: hid_t) -> Result<DType, Error> {
    use NumericType::*;
    let standard_types = [
        (I8, H5T_STD_I8LE, H5T_STD_I8BE),
        (U8, H5T_STD_U8LE, H5T_STD_U8BE),
        (I16, H5T_STD_I16LE, H5T_STD_I16BE),
        (U16, H5T_STD_U16LE, H5T_STD_U16BE),
        (I32, H5T_STD_I32LE, H5T_STD_I32BE),
        (U32, H5T_STD_U32LE, H5T_STD_U32BE),
        (I64, H5T_STD_I64LE, H5T_STD_I64BE),
        (U64, H5T_STD_U64LE, H5T_STD_U64BE),
        (F32, H5T_IEEE_F32LE, H5T_IEEE_F32BE),
        (F64, H5T_IEEE_F64LE, H5T_IEEE_F64BE),
    ];
    for (kind, little, big) in standard_types {
        for (standard, order) in [(little, ByteOrder::Little), (big, ByteOrder::Big)] {
            // SAFETY: Both IDs are live. Predefined IDs are initialized by H5open.
            let equal = check(unsafe { H5Tequal(datatype, *standard) }, "compare datatype")?;
            if equal > 0 {
                let byte_order = if kind.size() == 1 {
                    ByteOrder::NotApplicable
                } else {
                    order
                };
                return Ok(DType { kind, byte_order });
            }
        }
    }
    Err(Error::Unsupported("datatype"))
}

// The caller holds HDF5 and owns a live dataset creation property list.
unsafe fn filter_pipeline(properties: hid_t, item_size: usize) -> Result<Vec<Filter>, Error> {
    // SAFETY: This is a live dataset creation plist.
    let count = check(unsafe { H5Pget_nfilters(properties) }, "get filter count")?;
    if count > 32 {
        return Err(Error::Unsupported(
            "filter pipeline longer than its filter mask",
        ));
    }
    let mut filters = Vec::new();
    for index in 0..count as u32 {
        let mut length = 2;
        let mut parameters = [0; 2];
        // SAFETY: Parameters has capacity for the input length. The remaining
        // outputs are optional and the name capacity is zero.
        let filter = unsafe {
            H5Pget_filter2(
                properties,
                index,
                ptr::null_mut(),
                &mut length,
                parameters.as_mut_ptr(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let filter = match check(filter, "get filter")? {
            H5Z_FILTER_DEFLATE if length == 1 && parameters[0] <= 9 => Filter::Deflate {
                level: parameters[0],
            },
            H5Z_FILTER_SHUFFLE if length == 1 && parameters[0] as usize == item_size => {
                Filter::Shuffle
            }
            H5Z_FILTER_DEFLATE | H5Z_FILTER_SHUFFLE => {
                return Err(Error::Unsupported("filter parameters"))
            }
            id => return Err(Error::UnsupportedFilter(id)),
        };
        filters.push(filter);
    }
    if !matches!(
        filters.as_slice(),
        [] | [Filter::Shuffle]
            | [Filter::Deflate { .. }]
            | [Filter::Shuffle, Filter::Deflate { .. }]
    ) {
        return Err(Error::Unsupported("filter order or repeated filters"));
    }
    Ok(filters)
}

struct Id {
    raw: hid_t,
    close: unsafe extern "C" fn(hid_t) -> herr_t,
}

impl Id {
    fn new(
        raw: hid_t,
        close: unsafe extern "C" fn(hid_t) -> herr_t,
        operation: &'static str,
    ) -> Result<Self, Error> {
        if raw < 0 {
            return Err(Error::Hdf5(operation));
        }
        Ok(Self { raw, close })
    }
}

impl Drop for Id {
    fn drop(&mut self) {
        let _guard = HDF5.lock();
        // SAFETY: Id owns a successful HDF5 handle and stores its matching
        // close function. HDF5 calls, including closes, share the same lock.
        unsafe { (self.close)(self.raw) };
    }
}

fn check(status: herr_t, operation: &'static str) -> Result<herr_t, Error> {
    if status < 0 {
        Err(Error::Hdf5(operation))
    } else {
        Ok(status)
    }
}

fn with_hdf5<T>(operation: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    let _guard = HDF5.lock();
    // SAFETY: All calls here, including restoration, occur within the lock.
    unsafe {
        check(H5open(), "initialize HDF5")?;
        let mut callback = None;
        let mut data = ptr::null_mut();
        check(
            H5Eget_auto2(H5E_DEFAULT, &mut callback, &mut data),
            "get error callback",
        )?;
        let _restore = ErrorCallback { callback, data };
        check(
            H5Eset_auto2(H5E_DEFAULT, None, ptr::null_mut()),
            "disable error callback",
        )?;
        #[cfg(feature = "s3")]
        crate::s3::take_callback_error();
        let result = operation();
        #[cfg(feature = "s3")]
        if let Some(error) = crate::s3::take_callback_error() {
            if result.is_err() {
                return Err(error);
            }
        }
        result
    }
}

struct ErrorCallback {
    callback: H5E_auto2_t,
    data: *mut c_void,
}

impl Drop for ErrorCallback {
    fn drop(&mut self) {
        // SAFETY: This callback and data were retrieved from this error stack
        // under the same lock, which still covers this scope.
        unsafe { H5Eset_auto2(H5E_DEFAULT, self.callback, self.data) };
    }
}
