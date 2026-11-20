use std::collections::TryReserveError;
use std::io;

/// Errors returned by the reader.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid worker or concurrent-request count")]
    InvalidConcurrency,
    #[error("could not start read workers: {0}")]
    WorkerPool(#[from] rayon::ThreadPoolBuildError),
    #[error("read worker terminated before returning a result")]
    WorkerTerminated,
    #[cfg(feature = "s3")]
    #[error("invalid S3 object URL; use HTTPS or a loopback HTTP endpoint")]
    InvalidObjectUrl,
    #[cfg(feature = "s3")]
    #[error("invalid S3 range grouping limits")]
    InvalidRangeOptions,
    #[cfg(feature = "s3")]
    #[error("invalid S3 client timeouts; use finite positive durations")]
    InvalidS3Timeout,
    #[cfg(feature = "s3")]
    #[error("S3 CA bundle {path:?}: {reason}")]
    CaBundle {
        path: std::path::PathBuf,
        reason: &'static str,
    },
    #[cfg(feature = "s3")]
    #[error("S3 DNS resolution failed")]
    RemoteDns,
    #[cfg(feature = "s3")]
    #[error("S3 TLS failed: {0}")]
    RemoteTls(&'static str),
    #[cfg(feature = "s3")]
    #[error("S3 connection failed: {0}")]
    RemoteConnect(&'static str),
    #[cfg(feature = "s3")]
    #[error("S3 timeout during {0}")]
    RemoteTimeout(&'static str),
    #[cfg(feature = "s3")]
    #[error("S3 URL refresh failed or returned a different endpoint or path")]
    UrlRefresh,
    #[cfg(feature = "s3")]
    #[error("S3 range request failed (HTTP {0})")]
    RemoteStatus(u16),
    #[cfg(feature = "s3")]
    #[error("S3 HTTP transport failed")]
    RemoteTransport,
    #[cfg(feature = "s3")]
    #[error("S3 object no longer matches the identity pinned at open")]
    ObjectChanged,
    #[cfg(feature = "s3")]
    #[error("invalid S3 range response: {0}")]
    InvalidRangeResponse(&'static str),
    #[error("HDF5 operation failed: {0}")]
    Hdf5(&'static str),
    #[error("unsupported HDF5 feature: {0}")]
    Unsupported(&'static str),
    #[error("unsupported HDF5 filter (ID {0})")]
    UnsupportedFilter(i32),
    #[error("invalid file path or dataset name")]
    InvalidName,
    #[error("expected a selection of rank {expected}, got {actual}")]
    SelectionRank { expected: usize, actual: usize },
    #[error("invalid range {start}..{end} for axis {axis} of length {extent}")]
    SelectionBounds {
        axis: usize,
        start: u64,
        end: u64,
        extent: u64,
    },
    #[error("invalid stride, reduction axis or block size")]
    InvalidTransform,
    #[error("expected an output buffer of {expected} bytes, got {actual}")]
    OutputLength { expected: usize, actual: usize },
    #[error("size overflow computing {0}")]
    Overflow(&'static str),
    #[error("{kind} requires {bytes} bytes, exceeding the configured limit of {limit}")]
    Limit {
        kind: &'static str,
        bytes: usize,
        limit: usize,
    },
    #[error("invalid chunk: {0}")]
    InvalidChunk(&'static str),
    #[error("deflate decoding failed: {0}")]
    Deflate(#[from] flate2::DecompressError),
    #[error("buffer allocation failed: {0}")]
    Allocation(#[from] TryReserveError),
    #[error("file I/O failed: {0}")]
    Io(#[from] io::Error),
}

pub(crate) fn zeroed(bytes: usize) -> Result<Vec<u8>, Error> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(bytes)?;
    buffer.resize(bytes, 0);
    Ok(buffer)
}
