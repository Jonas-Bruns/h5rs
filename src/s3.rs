use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::io::Read;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use ureq::http::{header, HeaderValue, Response, Uri};
use ureq::{Agent, Body};

use crate::{error::zeroed, s3_client::transport_error, Error, S3Client};

const PAGE_SIZE: u64 = 64 * 1024;
const METADATA_PAGES: usize = 16;

thread_local! {
    // HDF5 reports VFD errors via an integer. Carry the Rust error back to the
    // calling metadata operation without exposing the possibly signed URL.
    static CALLBACK_ERROR: RefCell<Option<Error>> = const { RefCell::new(None) };
}

/// Settings for a read-only remote HDF5 handle.
#[derive(Clone)]
pub struct S3Options {
    /// Maximum chunks in a grouped range request.
    pub max_range_chunks: usize,
    /// Maximum bytes in a grouped range request, including gaps.
    pub max_range_bytes: usize,
    /// Maximum total unused bytes in each grouped range.
    pub max_gap_bytes: usize,
    /// Fetch and cache the first 64 KiB during the identity request.
    pub prefetch_header: bool,
    /// Build a bounded chunk address index when opening a chunked dataset.
    pub preload_chunk_index: bool,
    /// Share this client with other files at the same endpoint.
    pub client: Option<S3Client>,
    refresh_url: Option<Arc<dyn Fn() -> Result<String, Error> + Send + Sync>>,
}

impl Default for S3Options {
    fn default() -> Self {
        Self {
            max_range_chunks: 16,
            max_range_bytes: 2 * 1024 * 1024,
            max_gap_bytes: 0,
            prefetch_header: false,
            preload_chunk_index: false,
            client: None,
            refresh_url: None,
        }
    }
}

impl S3Options {
    /// Renew the URL once after a 401 or 403. The callback must sign the same
    /// object and must not call back into h5rs.
    pub fn with_url_refresh(
        mut self,
        refresh: impl Fn() -> Result<String, Error> + Send + Sync + 'static,
    ) -> Self {
        self.refresh_url = Some(Arc::new(refresh));
        self
    }
}

pub(crate) fn take_callback_error() -> Option<Error> {
    CALLBACK_ERROR.with(|error| error.borrow_mut().take())
}

pub(crate) struct Object {
    pub(crate) client: S3Client,
    url: Mutex<Uri>,
    refresh_lock: Mutex<()>,
    len: u64,
    etag: HeaderValue,
    metadata_pages: Mutex<VecDeque<(u64, Vec<u8>)>>,
    pub(crate) options: S3Options,
}

impl Object {
    pub(crate) fn open(url: &str, mut options: S3Options) -> Result<Self, Error> {
        if options.max_range_chunks == 0 || options.max_range_bytes == 0 {
            return Err(Error::InvalidRangeOptions);
        }
        let mut url = parse_url(url)?;
        let client = options.client.take().map_or_else(S3Client::shared, Ok)?;
        let agent = client.agent();
        let probe = 0..if options.prefetch_header {
            PAGE_SIZE
        } else {
            1
        };
        let mut response = match request(agent, &url, probe.clone(), None) {
            Err(Error::RemoteStatus(401 | 403)) if options.refresh_url.is_some() => {
                url = refresh_url(&url, &options)?;
                request(agent, &url, probe.clone(), None)?
            }
            other => other?,
        };
        let len = validate_range(&response, probe.clone(), None)?;
        let etag = response
            .headers()
            .get(header::ETAG)
            .filter(|tag| {
                let value = tag.as_bytes();
                value.len() >= 2 && value.starts_with(b"\"") && value.ends_with(b"\"")
            })
            .cloned()
            .ok_or(Error::InvalidRangeResponse("missing strong ETag"))?;
        let mut data = zeroed(len.min(probe.end) as usize)?;
        read_body(&mut response, &mut data)?;
        let mut metadata_pages = VecDeque::new();
        if options.prefetch_header {
            metadata_pages.push_back((0, data));
        }
        Ok(Self {
            client,
            url: Mutex::new(url),
            refresh_lock: Mutex::new(()),
            len,
            etag,
            metadata_pages: Mutex::new(metadata_pages),
            options,
        })
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    pub(crate) fn read_exact_at(&self, buffer: &mut [u8], start: u64) -> Result<(), Error> {
        if buffer.is_empty() {
            return Ok(());
        }
        let end = start
            .checked_add(buffer.len() as u64)
            .filter(|&end| end <= self.len)
            .ok_or(Error::InvalidRangeResponse(
                "requested range outside object",
            ))?;
        let range = start..end;
        let url = self.current_url();
        let agent = self.client.agent();
        let mut response = match request(agent, &url, range.clone(), Some(&self.etag)) {
            Err(Error::RemoteStatus(401 | 403)) if self.options.refresh_url.is_some() => {
                let next = self.renew_url(&url)?;
                request(agent, &next, range.clone(), Some(&self.etag))?
            }
            other => other?,
        };
        validate_range(&response, range, Some(self.len))?;
        if response.headers().get(header::ETAG) != Some(&self.etag) {
            return Err(Error::ObjectChanged);
        }
        read_body(&mut response, buffer)
    }

    pub(super) fn current_url(&self) -> Uri {
        self.url.lock().clone()
    }

    #[cfg(feature = "s3-async-io")]
    pub(super) fn can_refresh(&self) -> bool {
        self.options.refresh_url.is_some()
    }

    pub(super) fn renew_url(&self, previous: &Uri) -> Result<Uri, Error> {
        let _guard = self.refresh_lock.lock();
        let current = self.current_url();
        if current != *previous {
            return Ok(current);
        }
        let next = refresh_url(previous, &self.options)?;
        *self.url.lock() = next.clone();
        Ok(next)
    }

    #[cfg(feature = "s3-async-io")]
    pub(super) fn etag(&self) -> &HeaderValue {
        &self.etag
    }

    fn read_metadata(&self, mut buffer: &mut [u8], mut offset: u64) -> Result<(), Error> {
        if offset
            .checked_add(buffer.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(Error::InvalidRangeResponse("metadata range outside object"));
        }
        if buffer.len() as u64 > PAGE_SIZE {
            return self.read_exact_at(buffer, offset);
        }
        let mut pages = self.metadata_pages.lock();
        while !buffer.is_empty() {
            let start = offset / PAGE_SIZE * PAGE_SIZE;
            let page = match pages.iter().position(|(cached, _)| *cached == start) {
                Some(index) => pages.remove(index).expect("cache index exists").1,
                None => {
                    let size = (self.len - start).min(PAGE_SIZE) as usize;
                    let mut page = zeroed(size)?;
                    self.read_exact_at(&mut page, start)?;
                    if pages.len() == METADATA_PAGES {
                        pages.pop_back();
                    }
                    page
                }
            };
            let within = (offset - start) as usize;
            let count = buffer.len().min(page.len() - within);
            buffer[..count].copy_from_slice(&page[within..within + count]);
            pages.push_front((start, page));
            buffer = &mut buffer[count..];
            offset += count as u64;
        }
        Ok(())
    }
}

fn parse_url(url: &str) -> Result<Uri, Error> {
    let url: Uri = url.parse().map_err(|_| Error::InvalidObjectUrl)?;
    let scheme = url.scheme_str();
    let host = url.host().ok_or(Error::InvalidObjectUrl)?;
    let loopback = matches!(host, "localhost" | "127.0.0.1" | "[::1]");
    if scheme != Some("https") && !(scheme == Some("http") && loopback) {
        return Err(Error::InvalidObjectUrl);
    }
    Ok(url)
}

fn refresh_url(old: &Uri, options: &S3Options) -> Result<Uri, Error> {
    let callback = options.refresh_url.as_ref().ok_or(Error::UrlRefresh)?;
    let new = parse_url(&callback()?)?;
    if old.scheme() != new.scheme()
        || old.authority() != new.authority()
        || old.path() != new.path()
    {
        return Err(Error::UrlRefresh);
    }
    Ok(new)
}

fn request(
    agent: &Agent,
    url: &Uri,
    range: Range<u64>,
    etag: Option<&HeaderValue>,
) -> Result<Response<Body>, Error> {
    let request = agent
        .get(url.clone())
        .header(
            header::RANGE,
            format!("bytes={}-{}", range.start, range.end - 1),
        )
        .header(header::ACCEPT_ENCODING, "identity");
    let request = match etag {
        Some(etag) => request.header(header::IF_MATCH, etag.clone()),
        None => request,
    };
    let response = request.call().map_err(|err| match err {
        ureq::Error::StatusCode(412) if etag.is_some() => Error::ObjectChanged,
        ureq::Error::StatusCode(status) => Error::RemoteStatus(status),
        other => transport_error(&other),
    })?;
    match response.status().as_u16() {
        206 => Ok(response),
        200 => Err(Error::InvalidRangeResponse(
            "server ignored the Range header",
        )),
        status => Err(Error::RemoteStatus(status)),
    }
}

fn validate_range(
    response: &Response<Body>,
    range: Range<u64>,
    known_size: Option<u64>,
) -> Result<u64, Error> {
    validate_range_headers(response.headers(), range, known_size)
}

pub(super) fn validate_range_headers(
    headers: &ureq::http::HeaderMap,
    range: Range<u64>,
    known_size: Option<u64>,
) -> Result<u64, Error> {
    if headers
        .get(header::CONTENT_ENCODING)
        .is_some_and(|encoding| encoding != "identity")
    {
        return Err(Error::InvalidRangeResponse("encoded HTTP body"));
    }
    let parsed = headers
        .get(header::CONTENT_RANGE)
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.strip_prefix("bytes "))
        .and_then(|value| value.split_once('/'))
        .and_then(|(offsets, size)| {
            let (start, end) = offsets.split_once('-')?;
            Some((
                start.parse::<u64>().ok()?,
                end.parse::<u64>().ok()?,
                size.parse::<u64>().ok()?,
            ))
        });
    let Some((start, end, size)) = parsed else {
        return Err(Error::InvalidRangeResponse(
            "missing or malformed Content-Range",
        ));
    };
    if known_size.is_some_and(|known| known != size) {
        return Err(Error::ObjectChanged);
    }
    let expected_end = if known_size.is_none() {
        range.end.min(size)
    } else {
        range.end
    };
    if range.start != start || expected_end.checked_sub(1) != Some(end) || end >= size {
        return Err(Error::InvalidRangeResponse("unexpected range bounds"));
    }
    Ok(size)
}

fn read_body(response: &mut Response<Body>, buffer: &mut [u8]) -> Result<(), Error> {
    let mut body = response.body_mut().as_reader();
    let unreadable = |error: std::io::Error| match transport_error(&error) {
        Error::RemoteTransport => Error::InvalidRangeResponse("incomplete or unreadable HTTP body"),
        classified => classified,
    };
    body.read_exact(buffer).map_err(unreadable)?;
    if body.read(&mut [0]).map_err(unreadable)? != 0 {
        return Err(Error::InvalidRangeResponse("unexpected body length"));
    }
    Ok(())
}

#[no_mangle]
extern "C" fn h5rs_range_metadata_read(
    object: *const c_void,
    offset: u64,
    size: usize,
    buffer: *mut c_void,
) -> std::ffi::c_int {
    let result = std::panic::catch_unwind(|| {
        if size == 0 || size > isize::MAX as usize || buffer.is_null() || object.is_null() {
            return Err(Error::Hdf5("invalid range driver callback"));
        }
        // SAFETY: HDF5 passes a writable buffer of size bytes. The VFD borrows
        // the Arc allocation kept alive by the Rust file through H5Fclose.
        let object = unsafe { &*object.cast::<Object>() };
        let output = unsafe { std::slice::from_raw_parts_mut(buffer.cast::<u8>(), size) };
        object.read_metadata(output, offset)
    });
    match result {
        Ok(Ok(())) => 0,
        failure => {
            let error = failure.unwrap_or(Err(Error::WorkerTerminated)).unwrap_err();
            CALLBACK_ERROR.with(|slot| *slot.borrow_mut() = Some(error));
            -1
        }
    }
}
