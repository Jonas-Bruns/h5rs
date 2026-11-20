use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::io::Read;
use std::ops::Range;
use std::time::Duration;

use parking_lot::Mutex;
use ureq::http::{header, HeaderValue, Response, Uri};
use ureq::{Agent, Body};

use crate::{error::zeroed, Error};

const PAGE_SIZE: u64 = 64 * 1024;
const METADATA_PAGES: usize = 16;

thread_local! {
    // HDF5 reports VFD errors via an integer. Carry the Rust error back to the
    // calling metadata operation without exposing the possibly signed URL.
    static CALLBACK_ERROR: RefCell<Option<Error>> = const { RefCell::new(None) };
}

pub(crate) fn take_callback_error() -> Option<Error> {
    CALLBACK_ERROR.with(|error| error.borrow_mut().take())
}

pub(crate) struct Object {
    agent: Agent,
    url: Uri,
    len: u64,
    etag: HeaderValue,
    metadata_pages: Mutex<VecDeque<(u64, Vec<u8>)>>,
}

impl Object {
    pub(crate) fn open(url: &str, connections: usize) -> Result<Self, Error> {
        let url: Uri = url.parse().map_err(|_| Error::InvalidObjectUrl)?;
        let scheme = url.scheme_str();
        let host = url.host().ok_or(Error::InvalidObjectUrl)?;
        let loopback = matches!(host, "localhost" | "127.0.0.1" | "[::1]");
        if scheme != Some("https") && !(scheme == Some("http") && loopback) {
            return Err(Error::InvalidObjectUrl);
        }
        let agent: Agent = Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(30)))
            .max_redirects(0)
            .max_idle_connections(connections)
            .max_idle_connections_per_host(connections)
            .build()
            .into();
        let mut response = request(&agent, &url, 0..1, None)?;
        let len = validate_range(&response, 0..1, None)?;
        let etag = response
            .headers()
            .get(header::ETAG)
            .filter(|tag| {
                let value = tag.as_bytes();
                value.len() >= 2 && value.starts_with(b"\"") && value.ends_with(b"\"")
            })
            .cloned()
            .ok_or(Error::InvalidRangeResponse("missing strong ETag"))?;
        read_body(&mut response, &mut [0])?;
        Ok(Self {
            agent,
            url,
            len,
            etag,
            metadata_pages: Mutex::new(VecDeque::new()),
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
        let mut response = request(&self.agent, &self.url, range.clone(), Some(&self.etag))?;
        validate_range(&response, range, Some(self.len))?;
        if response.headers().get(header::ETAG) != Some(&self.etag) {
            return Err(Error::ObjectChanged);
        }
        read_body(&mut response, buffer)
    }

    fn read_metadata(&self, mut buffer: &mut [u8], mut offset: u64) -> Result<(), Error> {
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
        _ => Error::RemoteTransport,
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
    if response
        .headers()
        .get(header::CONTENT_ENCODING)
        .is_some_and(|encoding| encoding != "identity")
    {
        return Err(Error::InvalidRangeResponse("encoded HTTP body"));
    }
    let parsed = response
        .headers()
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
    if range.start != start || range.end - 1 != end || end >= size {
        return Err(Error::InvalidRangeResponse("unexpected range bounds"));
    }
    Ok(size)
}

fn read_body(response: &mut Response<Body>, buffer: &mut [u8]) -> Result<(), Error> {
    let mut body = response.body_mut().as_reader();
    let unreadable = |_| Error::InvalidRangeResponse("incomplete or unreadable HTTP body");
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
