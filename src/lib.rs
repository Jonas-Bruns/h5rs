//! Parallel read-only access to numeric HDF5 datasets.
//!
//! Dataset handles can be shared between threads. Data I/O and filter decoding
//! run on a bounded worker pool outside the process-wide HDF5 metadata lock.

mod cache;
mod error;
mod filters;
mod metadata;
mod pool;
mod raw;
mod region;
#[cfg(feature = "s3")]
mod s3;
#[cfg(feature = "s3-async-io")]
mod s3_async;
mod transform;

use std::borrow::Cow;
use std::ffi::CString;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;

pub use error::Error;
use metadata::ChunkLocation;
pub use pool::ReadPool;
use raw::RawFile;
use region::{Layout, Selection};
#[cfg(feature = "s3")]
pub use s3::S3Options;

/// Primitive numeric types with standard HDF5 binary representations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl NumericType {
    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// Byte order in the file and in output buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteOrder {
    Little,
    Big,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DType {
    pub kind: NumericType,
    pub byte_order: ByteOrder,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        self.kind.size()
    }
}

/// A filter in the dataset's write pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Filter {
    Shuffle,
    Deflate { level: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetMetadata {
    pub shape: Vec<u64>,
    /// Chunk dimensions, or an empty vector for contiguous and compact storage.
    pub chunks: Vec<u64>,
    pub dtype: DType,
    /// Filters in write order. Individual chunks may skip filters.
    pub filters: Vec<Filter>,
    /// One fill element in the dataset's byte order.
    pub fill_value: Vec<u8>,
}

/// Maximum allocation sizes per call. A shared pool also limits concurrent calls.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_output_bytes: usize,
    /// Applies independently to stored and decoded chunk sizes.
    pub max_chunk_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_output_bytes: 512 * 1024 * 1024,
            max_chunk_bytes: 64 * 1024 * 1024,
        }
    }
}

/// A rectangular selection in row-major order, with the last axis contiguous.
#[derive(Debug)]
pub struct Region {
    pub shape: Vec<u64>,
    pub dtype: DType,
    /// Packed elements in the original byte order.
    pub bytes: Vec<u8>,
}

/// Shared read-only access to an immutable HDF5 file.
#[derive(Clone)]
pub struct H5File(Arc<FileInner>);

struct FileInner {
    // Drop the HDF5 handle before releasing the remote object it may borrow.
    metadata: metadata::File,
    raw: RawFile,
    limits: Limits,
    pool: ReadPool,
}

impl H5File {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::open_with_limits(path, Limits::default())
    }

    /// The file must remain unmodified and at the same path during use.
    pub fn open_with_limits(path: impl AsRef<Path>, limits: Limits) -> Result<Self, Error> {
        Self::open_with_pool(path, limits, ReadPool::shared()?)
    }

    /// Open a local file using a caller-provided shared worker pool.
    pub fn open_with_pool(
        path: impl AsRef<Path>,
        limits: Limits,
        pool: ReadPool,
    ) -> Result<Self, Error> {
        let path = std::path::absolute(path)?;
        #[cfg(unix)]
        let name = {
            use std::os::unix::ffi::OsStrExt;
            path.as_os_str().as_bytes()
        };
        #[cfg(windows)]
        let name = path.to_str().ok_or(Error::InvalidName)?.as_bytes();
        let name = CString::new(name).map_err(|_| Error::InvalidName)?;
        if !std::fs::metadata(&path)?.is_file() {
            return Err(Error::Unsupported("non-regular file"));
        }
        let raw = RawFile::open(&path)?;
        let metadata = metadata::File::open(&name)?;
        Ok(Self(Arc::new(FileInner {
            metadata,
            raw,
            limits,
            pool,
        })))
    }

    /// Open a local file away from the caller's async executor.
    pub async fn open_async(path: impl Into<PathBuf>) -> Result<Self, Error> {
        let path = path.into();
        let pool = ReadPool::shared()?;
        let worker_pool = pool.clone();
        pool.submit(move || Self::open_with_pool(path, Limits::default(), worker_pool))
            .await
    }

    /// Open an S3 object using a public HTTPS URL or a presigned GET URL.
    ///
    /// Signed URLs must allow Range and If-Match headers to vary. The initial
    /// range response must provide a strong ETag for subsequent reads.
    #[cfg(feature = "s3")]
    pub fn open_s3(url: &str) -> Result<Self, Error> {
        Self::open_s3_with_defaults(url, S3Options::default())
    }

    /// Open a remote file with the default worker pool and memory limits.
    #[cfg(feature = "s3")]
    pub fn open_s3_with_defaults(url: &str, options: S3Options) -> Result<Self, Error> {
        Self::open_s3_with_options(url, Limits::default(), ReadPool::shared()?, options)
    }

    /// Open an S3 object using a caller-provided pool and memory limits.
    #[cfg(feature = "s3")]
    pub fn open_s3_with_pool(url: &str, limits: Limits, pool: ReadPool) -> Result<Self, Error> {
        Self::open_s3_with_options(url, limits, pool, S3Options::default())
    }

    /// Open a remote file with configurable range grouping and URL renewal.
    #[cfg(feature = "s3")]
    pub fn open_s3_with_options(
        url: &str,
        limits: Limits,
        pool: ReadPool,
        options: S3Options,
    ) -> Result<Self, Error> {
        let object = Arc::new(s3::Object::open(url, pool.threads(), options)?);
        // SAFETY: raw stores the Arc for at least as long as the HDF5 file.
        // The dataset handle is likewise dropped before its file reference.
        let metadata = unsafe { metadata::File::open_s3(&object)? };
        Ok(Self(Arc::new(FileInner {
            metadata,
            raw: RawFile::Remote(object),
            limits,
            pool,
        })))
    }

    /// Open an S3 object away from the caller's async executor.
    #[cfg(feature = "s3")]
    pub async fn open_s3_async(url: &str) -> Result<Self, Error> {
        let url = url.to_owned();
        let pool = ReadPool::shared()?;
        let worker_pool = pool.clone();
        pool.submit(move || Self::open_s3_with_pool(&url, Limits::default(), worker_pool))
            .await
    }

    pub fn dataset(&self, name: &str) -> Result<Dataset, Error> {
        let name = CString::new(name).map_err(|_| Error::InvalidName)?;
        let (handle, metadata) = self.0.metadata.dataset(&name)?;
        let storage_shape = match handle.storage {
            metadata::Storage::Chunked => &metadata.chunks,
            _ => &metadata.shape,
        };
        let chunk = Layout::new(storage_shape, metadata.dtype.size(), "dataset storage")?;
        if !matches!(handle.storage, metadata::Storage::Contiguous(_)) {
            check_limit(chunk.len, self.0.limits.max_chunk_bytes, "decoded chunk")?;
        }
        let dataset = Dataset {
            handle: Arc::new(handle),
            file: self.clone(),
            metadata: Arc::new(metadata),
            chunk,
            cache: None,
        };
        #[cfg(feature = "s3")]
        if let RawFile::Remote(object) = &self.0.raw {
            if object.options.preload_chunk_index
                && matches!(dataset.handle.storage, metadata::Storage::Chunked)
            {
                match dataset.handle.prepare_s3_index() {
                    Ok(()) | Err(Error::Limit { .. }) => {}
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(dataset)
    }

    /// Open an S3 dataset and prebuild its read-only chunk address index in RAM.
    /// The index lives for as long as this dataset or any of its clones does.
    /// Use ordinary dataset access when the index exceeds the memory limit.
    #[cfg(feature = "s3")]
    pub fn dataset_with_s3_index(&self, name: &str) -> Result<Dataset, Error> {
        if !matches!(&self.0.raw, RawFile::Remote(_)) {
            return Err(Error::Unsupported("S3 index requires an S3 object"));
        }
        let dataset = self.dataset(name)?;
        if matches!(dataset.handle.storage, metadata::Storage::Chunked) {
            dataset.handle.prepare_s3_index()?;
        }
        Ok(dataset)
    }

    /// Load dataset metadata away from the caller's async executor.
    pub async fn dataset_async(&self, name: &str) -> Result<Dataset, Error> {
        let name = name.to_owned();
        let file = self.clone();
        self.0.pool.submit(move || file.dataset(&name)).await
    }
}

/// A 1D, 2D or 3D dataset handle. It keeps its file open and is safe to share
/// between threads.
#[derive(Clone)]
pub struct Dataset {
    handle: Arc<metadata::Dataset>,
    file: H5File,
    metadata: Arc<DatasetMetadata>,
    chunk: Layout,
    cache: Option<Arc<parking_lot::Mutex<cache::ChunkCache>>>,
}

impl Dataset {
    /// Return a handle with an optional decoded-chunk cache, shared by its
    /// clones. The budget applies to resident decoded byte buffers per cache.
    pub fn with_chunk_cache(mut self, capacity_bytes: usize) -> Self {
        self.cache = (capacity_bytes > 0
            && self.chunk.len <= capacity_bytes
            && matches!(self.handle.storage, metadata::Storage::Chunked))
        .then(|| {
            Arc::new(parking_lot::Mutex::new(cache::ChunkCache::new(
                capacity_bytes,
            )))
        });
        self
    }

    fn cache_chunk(&self, origin: [u64; 3], data: Vec<u8>) {
        if let Some(cache) = &self.cache {
            cache.lock().insert(origin, data);
        }
    }

    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Read one half-open, step-1 range per axis.
    pub fn read_region(&self, ranges: &[Range<u64>]) -> Result<Region, Error> {
        let selection = self.selection(ranges)?;
        self.file.0.pool.run(|| self.read_owned(selection))
    }

    /// Read a packed strided region. Negative steps traverse the corresponding
    /// half-open range from its last element. The bounding region is read first.
    pub fn read_strided(&self, ranges: &[Range<u64>], steps: &[isize]) -> Result<Region, Error> {
        self.read_transformed(ranges, steps, false)
    }

    /// Read a packed strided region, converting the selected values to float32.
    pub fn read_f32(&self, ranges: &[Range<u64>], steps: &[isize]) -> Result<Region, Error> {
        self.read_transformed(ranges, steps, true)
    }

    fn read_transformed(
        &self,
        ranges: &[Range<u64>],
        steps: &[isize],
        cast: bool,
    ) -> Result<Region, Error> {
        if ranges.len() != steps.len() || steps.contains(&0) {
            return Err(Error::InvalidTransform);
        }
        let selection = self.selection(ranges)?;
        self.file.0.pool.run(|| {
            let region = transform::stride(self.read_owned(selection)?, steps)?;
            if !cast
                || region.dtype
                    == (DType {
                        kind: NumericType::F32,
                        byte_order: ByteOrder::Little,
                    })
            {
                return Ok(region);
            }
            let size = Layout::new(&region.shape, 4, "float32 output")?.len;
            check_limit(size, self.file.0.limits.max_output_bytes, "float32 output")?;
            Ok(transform::f32(region, error::zeroed(size)?))
        })
    }

    /// Reduce fixed-size bins along one axis. The last output axis contains
    /// [minimum, maximum] in the original datatype. NaNs are ignored unless
    /// every value in a bin is NaN.
    pub fn read_min_max(
        &self,
        ranges: &[Range<u64>],
        axis: usize,
        block: usize,
    ) -> Result<Region, Error> {
        let selection = self.selection(ranges)?;
        let mut shape = selection.shape.clone();
        if axis >= shape.len() || block == 0 {
            return Err(Error::InvalidTransform);
        }
        shape[axis] = shape[axis].div_ceil(block as u64);
        let size = Layout::new(&shape, self.metadata.dtype.size() * 2, "min/max output")?.len;
        check_limit(size, self.file.0.limits.max_output_bytes, "min/max output")?;
        shape.push(2);
        self.file.0.pool.run(|| {
            let input = self.read_owned(selection)?;
            let output = Region {
                shape,
                dtype: input.dtype,
                bytes: error::zeroed(size)?,
            };
            Ok(transform::min_max(input, axis, block, output))
        })
    }

    /// Read a region without blocking the caller's async executor.
    ///
    /// Dropping the future discards the result. I/O already running on the
    /// worker pool may finish before its admission slot is released.
    pub async fn read_region_async(&self, ranges: &[Range<u64>]) -> Result<Region, Error> {
        let selection = self.selection(ranges)?;
        let dataset = self.clone();
        self.file
            .0
            .pool
            .submit(move || dataset.read_owned(selection))
            .await
    }

    fn read_owned(&self, selection: Selection) -> Result<Region, Error> {
        let mut bytes = error::zeroed(selection.output.len)?;
        self.read_selection(&selection, &mut bytes)?;
        Ok(Region {
            shape: selection.shape,
            dtype: self.metadata.dtype,
            bytes,
        })
    }

    /// Read into an exact-length caller-provided buffer. On error the buffer may
    /// be partially written. The configured output limit applies to this API.
    pub fn read_region_into(&self, ranges: &[Range<u64>], output: &mut [u8]) -> Result<(), Error> {
        let selection = self.selection(ranges)?;
        if selection.output.len != output.len() {
            return Err(Error::OutputLength {
                expected: selection.output.len,
                actual: output.len(),
            });
        }
        self.file
            .0
            .pool
            .run(|| self.read_selection(&selection, output))
    }

    fn selection(&self, ranges: &[Range<u64>]) -> Result<Selection, Error> {
        let selection = Selection::new(ranges, &self.metadata.shape, self.metadata.dtype.size())?;
        check_limit(
            selection.output.len,
            self.file.0.limits.max_output_bytes,
            "output",
        )?;
        Ok(selection)
    }

    fn read_selection(&self, selection: &Selection, output: &mut [u8]) -> Result<(), Error> {
        if output.is_empty() {
            return Ok(());
        }
        let fill = &self.metadata.fill_value;
        let shape = region::padded(&self.metadata.shape, 1);
        match self.handle.storage {
            metadata::Storage::Contiguous(Some(offset)) => {
                return selection.read_contiguous(
                    self.chunk,
                    shape,
                    fill.len(),
                    output,
                    |relative, bytes| {
                        let address = offset
                            .checked_add(relative as u64)
                            .ok_or(Error::Overflow("file offset"))?;
                        self.file.0.raw.read_exact_at(bytes, address)
                    },
                );
            }
            metadata::Storage::Contiguous(None) => {
                for element in output.chunks_exact_mut(fill.len()) {
                    element.copy_from_slice(fill);
                }
                return Ok(());
            }
            metadata::Storage::Compact => {
                if output.len() == self.chunk.len {
                    return self.handle.read_compact(output);
                }
                let mut data = error::zeroed(self.chunk.len)?;
                self.handle.read_compact(&mut data)?;
                selection.copy_chunk([0; 3], shape, self.chunk, Some(&data), fill, output);
                return Ok(());
            }
            metadata::Storage::Chunked => {}
        }
        let chunk_shape = region::padded(&self.metadata.chunks, 1);
        let rank = self.metadata.shape.len();
        let file = &self.file.0;
        let locations = match &file.raw {
            RawFile::Local(_) => self.handle.location_index(),
            #[cfg(feature = "s3")]
            RawFile::Remote(_) => self.handle.prepared_index(),
        };
        let threads = file.pool.threads();
        #[cfg(feature = "s3-async-io")]
        let threads = match (&file.raw, file.pool.s3_transport()) {
            (RawFile::Remote(_), Some(transport)) => threads.max(transport.limit),
            _ => threads,
        };
        let unfiltered = self.metadata.filters.is_empty();
        let span_limits = match &file.raw {
            RawFile::Local(_) if unfiltered => Some(SpanLimits {
                chunks: 4,
                bytes: 512 * 1024,
                gaps: 0,
            }),
            RawFile::Local(_) => None,
            #[cfg(feature = "s3")]
            RawFile::Remote(object) => Some(SpanLimits {
                chunks: object.options.max_range_chunks,
                bytes: object.options.max_range_bytes,
                gaps: object
                    .options
                    .max_gap_bytes
                    .min(object.options.max_range_bytes),
            }),
        };
        let batch_chunks = threads
            .saturating_mul(span_limits.map_or(1, |limits| limits.chunks))
            .min(65_536);
        // Bound the sum across the batch, including staged reads and decoded
        // data. A large filtered chunk can always use the single-chunk path.
        let budget = file
            .limits
            .max_chunk_bytes
            .saturating_mul(2)
            .saturating_add(1);
        let mut batch = Vec::with_capacity(batch_chunks);
        let mut used = 0_usize;
        for origin in selection.chunk_origins(chunk_shape) {
            if let Some(data) = self
                .cache
                .as_ref()
                .and_then(|cache| cache.lock().get(origin))
            {
                selection.copy_chunk(
                    origin,
                    chunk_shape,
                    self.chunk,
                    Some(data.as_slice()),
                    fill,
                    output,
                );
                continue;
            }
            let location = match locations {
                Some(index) => index.location(origin),
                None => self.handle.chunk_location(&origin[3 - rank..])?,
            };
            let size = location.as_ref().map_or(Ok(0), |location| {
                file.raw
                    .validate_chunk(location, file.limits.max_chunk_bytes)
            })?;
            let reserve = if location.is_none() {
                0
            } else if unfiltered {
                if size != self.chunk.len {
                    return Err(Error::InvalidChunk("unexpected decoded size"));
                }
                size
            } else if span_limits.is_some() {
                size.saturating_add(self.chunk.len.saturating_mul(2))
                    .saturating_add(1)
            } else {
                size.max(self.chunk.len)
                    .saturating_add(self.chunk.len)
                    .saturating_add(1)
            };
            let reserve = if location.is_some() {
                reserve.saturating_add(span_limits.map_or(0, |limits| limits.gaps))
            } else {
                reserve
            };
            if !batch.is_empty() && (batch.len() == batch_chunks || reserve > budget - used) {
                self.read_batch(&mut batch, selection, chunk_shape, output, span_limits)?;
                batch.clear();
                used = 0;
            }
            batch.push(ChunkJob {
                origin,
                location,
                size,
            });
            if reserve > budget {
                self.read_batch(&mut batch, selection, chunk_shape, output, None)?;
                batch.clear();
            } else {
                used += reserve;
            }
        }
        self.read_batch(&mut batch, selection, chunk_shape, output, span_limits)
    }

    fn read_batch(
        &self,
        batch: &mut [ChunkJob],
        selection: &Selection,
        chunk_shape: [u64; 3],
        output: &mut [u8],
        span_limits: Option<SpanLimits>,
    ) -> Result<(), Error> {
        let mut copy = |job: &ChunkJob, data: Option<&[u8]>| {
            selection.copy_chunk(
                job.origin,
                chunk_shape,
                self.chunk,
                data,
                &self.metadata.fill_value,
                output,
            );
        };
        #[cfg(feature = "s3-async-io")]
        if let (RawFile::Remote(_), Some(_)) = (&self.file.0.raw, self.file.0.pool.s3_transport()) {
            return self.read_grouped_batch(
                batch,
                span_limits.unwrap_or(SpanLimits {
                    chunks: 1,
                    bytes: 1,
                    gaps: 0,
                }),
                copy,
            );
        }
        if let Some(limits) = span_limits.filter(|_| batch.len() > 1) {
            return self.read_grouped_batch(batch, limits, copy);
        }
        match batch {
            [] => {}
            [job] => {
                let data = self.load_chunk(job)?;
                copy(job, data.as_deref());
                if let Some(data) = data {
                    self.cache_chunk(job.origin, data);
                }
            }
            _ => {
                let decoded: Result<Vec<_>, _> = batch
                    .par_iter()
                    .map(|job| self.load_chunk(job).map(|data| (job, data)))
                    .collect();
                for (job, data) in decoded? {
                    copy(job, data.as_deref());
                    if let Some(data) = data {
                        self.cache_chunk(job.origin, data);
                    }
                }
            }
        }
        Ok(())
    }

    fn read_grouped_batch(
        &self,
        batch: &mut [ChunkJob],
        limits: SpanLimits,
        mut copy: impl FnMut(&ChunkJob, Option<&[u8]>),
    ) -> Result<(), Error> {
        batch.sort_unstable_by_key(|job| job.location.map(|location| location.offset));
        let (missing, allocated) =
            batch.split_at(batch.partition_point(|job| job.location.is_none()));
        for job in missing {
            copy(job, None);
        }
        let spans = read_spans(allocated, limits);
        let load = |span: &ReadSpan<'_>| {
            let mut bytes = error::zeroed(span.len)?;
            self.file.0.raw.read_exact_at(&mut bytes, span.offset)?;
            Ok(bytes)
        };
        let load_all = || {
            #[cfg(feature = "s3-async-io")]
            if let (RawFile::Remote(object), Some(transport)) =
                (&self.file.0.raw, self.file.0.pool.s3_transport())
            {
                let ranges = spans.iter().map(|span| (span.offset, span.len)).collect();
                let buffers = transport.read(object, ranges)?;
                return Ok::<_, Error>(spans.iter().zip(buffers).collect::<Vec<_>>());
            }
            spans
                .par_iter()
                .map(|span| load(span).map(|bytes| (span, bytes)))
                .collect::<Result<Vec<_>, Error>>()
        };
        if !self.metadata.filters.is_empty() {
            let buffers = load_all()?;
            let decoded = buffers
                .par_iter()
                .flat_map(|(span, bytes)| {
                    span.jobs
                        .par_iter()
                        .map(move |job| (job, span.chunk_bytes(job, bytes)))
                })
                .map(|(job, bytes)| {
                    self.decode_chunk(job, Cow::Borrowed(bytes))
                        .map(|data| (job, data))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            for (job, data) in decoded {
                copy(job, Some(&data));
                if self.cache.is_some() {
                    self.cache_chunk(job.origin, data.into_owned());
                }
            }
            return Ok(());
        }
        let mut copy_span = |span: &ReadSpan<'_>, bytes: &[u8]| {
            for job in span.jobs {
                let data = span.chunk_bytes(job, bytes);
                copy(job, Some(data));
                if self.cache.is_some() {
                    self.cache_chunk(job.origin, data.to_vec());
                }
            }
        };
        #[cfg(feature = "s3-async-io")]
        if let (RawFile::Remote(_), Some(_)) = (&self.file.0.raw, self.file.0.pool.s3_transport()) {
            for (span, bytes) in load_all()? {
                copy_span(span, &bytes);
            }
            return Ok(());
        }
        match spans.as_slice() {
            [] => {}
            [span] => copy_span(span, &load(span)?),
            _ => {
                let buffers = load_all()?;
                for (span, bytes) in buffers {
                    copy_span(span, &bytes);
                }
            }
        }
        Ok(())
    }

    fn load_chunk(&self, job: &ChunkJob) -> Result<Option<Vec<u8>>, Error> {
        let Some(location) = &job.location else {
            return Ok(None);
        };
        let mut data = error::zeroed(job.size)?;
        self.file.0.raw.read_exact_at(&mut data, location.offset)?;
        self.decode_chunk(job, Cow::Owned(data))
            .map(|data| Some(data.into_owned()))
    }

    fn decode_chunk<'a>(
        &self,
        job: &ChunkJob,
        data: Cow<'a, [u8]>,
    ) -> Result<Cow<'a, [u8]>, Error> {
        filters::decode(
            data,
            job.location.expect("allocated jobs only").filter_mask,
            &self.metadata.filters,
            self.chunk.len,
            self.metadata.dtype.size(),
        )
    }
}

struct ChunkJob {
    origin: [u64; 3],
    location: Option<ChunkLocation>,
    size: usize,
}

#[derive(Copy, Clone)]
struct SpanLimits {
    chunks: usize,
    bytes: usize,
    gaps: usize,
}

struct ReadSpan<'a> {
    jobs: &'a [ChunkJob],
    offset: u64,
    len: usize,
}

impl ReadSpan<'_> {
    fn chunk_bytes<'a>(&self, job: &ChunkJob, buffer: &'a [u8]) -> &'a [u8] {
        let relative = (job.location.expect("allocated jobs only").offset - self.offset) as usize;
        &buffer[relative..relative + job.size]
    }
}

fn read_spans(jobs: &[ChunkJob], limits: SpanLimits) -> Vec<ReadSpan<'_>> {
    let mut spans = Vec::new();
    let mut start = 0;
    let mut len = 0;
    let mut offset = 0;
    let mut gaps = 0;
    for (index, job) in jobs.iter().enumerate() {
        let address = job.location.expect("allocated jobs only").offset;
        let gap = address
            .checked_sub(offset + len as u64)
            .and_then(|gap| usize::try_from(gap).ok());
        if len > 0
            && (index - start == limits.chunks
                || gap.is_none_or(|gap| {
                    gap > limits.gaps - gaps
                        || gap.saturating_add(job.size) > limits.bytes.saturating_sub(len)
                }))
        {
            spans.push(ReadSpan {
                jobs: &jobs[start..index],
                offset,
                len,
            });
            start = index;
            len = 0;
            gaps = 0;
        }
        if len == 0 {
            offset = address;
        } else {
            let gap = gap.expect("grouped spans do not overlap");
            gaps += gap;
            len += gap;
        }
        len += job.size;
    }
    if len > 0 {
        spans.push(ReadSpan {
            jobs: &jobs[start..],
            offset,
            len,
        });
    }
    spans
}

fn check_limit(bytes: usize, limit: usize, kind: &'static str) -> Result<(), Error> {
    if bytes > limit {
        return Err(Error::Limit { kind, bytes, limit });
    }
    Ok(())
}
