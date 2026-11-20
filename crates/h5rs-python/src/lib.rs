use std::ops::Range;
use std::path::PathBuf;
use std::time::Duration;

use h5rs_core::{
    ByteOrder, DType, Error, H5File, Limits, NumericType, ReadPool, Region, S3Client,
    S3ClientConfig, S3Options,
};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyIOError, PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PySliceMethods, PyTuple};
use rayon::prelude::*;

pyo3::create_exception!(h5rs, S3Error, PyIOError);

#[pyclass(name = "S3Client", module = "h5rs", frozen)]
struct PythonS3Client(S3Client);

#[pymethods]
impl PythonS3Client {
    #[new]
    #[pyo3(signature = (*, ca_bundle=None, connect_timeout=30.0, request_timeout=30.0))]
    fn new(
        py: Python<'_>,
        ca_bundle: Option<PathBuf>,
        connect_timeout: f64,
        request_timeout: f64,
    ) -> PyResult<Self> {
        let config = s3_config(ca_bundle, Some(connect_timeout), Some(request_timeout))?;
        py.detach(|| S3Client::new(config))
            .map(Self)
            .map_err(python_error)
    }
}

fn s3_config(
    ca_bundle: Option<PathBuf>,
    connect_timeout: Option<f64>,
    request_timeout: Option<f64>,
) -> PyResult<S3ClientConfig> {
    let defaults = S3ClientConfig::default();
    let duration = |seconds: Option<f64>, default: Duration| match seconds {
        Some(seconds) => {
            Duration::try_from_secs_f64(seconds).map_err(|_| python_error(Error::InvalidS3Timeout))
        }
        None => Ok(default),
    };
    Ok(S3ClientConfig {
        ca_bundle,
        connect_timeout: duration(connect_timeout, defaults.connect_timeout)?,
        request_timeout: duration(request_timeout, defaults.request_timeout)?,
    })
}

#[pyclass(name = "ReadPool", module = "h5rs", frozen)]
struct PythonPool(ReadPool);

#[pymethods]
impl PythonPool {
    #[new]
    #[pyo3(signature = (threads, concurrent_requests, *, max_s3_gets=None))]
    fn new(
        threads: usize,
        concurrent_requests: usize,
        max_s3_gets: Option<usize>,
    ) -> PyResult<Self> {
        let pool = match max_s3_gets {
            Some(limit) => ReadPool::with_s3_io(threads, concurrent_requests, limit),
            None => ReadPool::new(threads, concurrent_requests),
        };
        pool.map(Self).map_err(python_error)
    }
}

#[pyclass(name = "File", module = "h5rs", frozen)]
struct PythonFile(H5File);

#[pymethods]
impl PythonFile {
    #[new]
    #[pyo3(signature = (path, pool=None))]
    fn new(py: Python<'_>, path: PathBuf, pool: Option<PyRef<'_, PythonPool>>) -> PyResult<Self> {
        let pool = pool.map(|pool| pool.0.clone());
        py.detach(move || match pool {
            Some(pool) => H5File::open_with_pool(path, Limits::default(), pool),
            None => H5File::open(path),
        })
        .map(Self)
        .map_err(python_error)
    }

    /// Open an immutable S3 object using a public or presigned HTTPS GET URL.
    #[staticmethod]
    #[pyo3(signature = (url, pool=None, *, refresh_url=None, max_range_chunks=None, max_range_bytes=None, max_gap_bytes=0, prefetch_header=false, preload_chunk_index=false, client=None, ca_bundle=None, connect_timeout=None, request_timeout=None))]
    #[allow(clippy::too_many_arguments)]
    fn open_s3(
        py: Python<'_>,
        url: &str,
        pool: Option<PyRef<'_, PythonPool>>,
        refresh_url: Option<Py<PyAny>>,
        max_range_chunks: Option<usize>,
        max_range_bytes: Option<usize>,
        max_gap_bytes: usize,
        prefetch_header: bool,
        preload_chunk_index: bool,
        client: Option<PyRef<'_, PythonS3Client>>,
        ca_bundle: Option<PathBuf>,
        connect_timeout: Option<f64>,
        request_timeout: Option<f64>,
    ) -> PyResult<Self> {
        let pool = pool.map(|pool| pool.0.clone());
        let mut options = S3Options::default();
        options.max_range_chunks = max_range_chunks.unwrap_or(options.max_range_chunks);
        options.max_range_bytes = max_range_bytes.unwrap_or(options.max_range_bytes);
        options.max_gap_bytes = max_gap_bytes;
        options.prefetch_header = prefetch_header;
        options.preload_chunk_index = preload_chunk_index;
        let config =
            if ca_bundle.is_some() || connect_timeout.is_some() || request_timeout.is_some() {
                if client.is_some() {
                    return Err(PyValueError::new_err(
                        "configure CA and timeouts on the supplied S3Client",
                    ));
                }
                Some(s3_config(ca_bundle, connect_timeout, request_timeout)?)
            } else {
                None
            };
        options.client = client.map(|client| client.0.clone());
        if let Some(refresh) = refresh_url {
            options = options.with_url_refresh(move || {
                Python::attach(|py| refresh.call0(py).and_then(|url| url.extract::<String>(py)))
                    .map_err(|_| Error::UrlRefresh)
            });
        }
        py.detach(move || {
            if let Some(config) = config {
                options.client = Some(S3Client::new(config)?);
            }
            open_remote(url, pool, options)
        })
        .map(Self)
        .map_err(python_error)
    }

    /// Open an S3 dataset and build its chunk index in RAM.
    fn dataset_with_s3_index(&self, py: Python<'_>, name: &str) -> PyResult<PythonDataset> {
        py.detach(|| self.0.dataset_with_s3_index(name))
            .map(PythonDataset)
            .map_err(python_error)
    }

    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<PythonDataset> {
        py.detach(|| self.0.dataset(name))
            .map(PythonDataset)
            .map_err(python_error)
    }
}

#[pyclass(name = "Dataset", module = "h5rs", frozen)]
struct PythonDataset(h5rs_core::Dataset);

#[pymethods]
impl PythonDataset {
    fn with_chunk_cache(&self, capacity_bytes: usize) -> Self {
        Self(self.0.clone().with_chunk_cache(capacity_bytes))
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.metadata().shape.iter().copied())
    }

    #[getter]
    fn chunks<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let chunks = &self.0.metadata().chunks;
        if chunks.is_empty() {
            return Ok(None);
        }
        PyTuple::new(py, chunks.iter().copied()).map(Some)
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("numpy")?
            .getattr("dtype")?
            .call1((numpy_type(self.0.metadata().dtype),))
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        selection: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.read(py, selection, None)
    }

    #[pyo3(signature = (selection, *, dtype=None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        selection: &Bound<'py, PyAny>,
        dtype: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (ranges, steps) = parse_selection(selection, &self.0.metadata().shape)?;
        let result = py
            .detach(|| {
                if dtype == Some("float32") {
                    self.0.read_f32(&ranges, &steps)
                } else {
                    self.0.read_strided(&ranges, &steps)
                }
            })
            .map_err(python_error)?;
        let array = numpy_region(py, result)?;
        match dtype {
            Some(dtype) if dtype != "float32" => array.call_method1("astype", (dtype,)),
            _ => Ok(array),
        }
    }

    #[pyo3(signature = (selection, block, *, axis=0))]
    fn read_minmax<'py>(
        &self,
        py: Python<'py>,
        selection: &Bound<'py, PyAny>,
        block: usize,
        axis: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (ranges, steps) = parse_selection(selection, &self.0.metadata().shape)?;
        if steps.iter().any(|step| *step != 1) {
            return Err(PyValueError::new_err("min/max requires step-1 slices"));
        }
        let result = py
            .detach(|| self.0.read_min_max(&ranges, axis, block))
            .map_err(python_error)?;
        numpy_region(py, result)
    }

    #[pyo3(signature = (selection, block, *, axis=0))]
    fn read_mean<'py>(
        &self,
        py: Python<'py>,
        selection: &Bound<'py, PyAny>,
        block: usize,
        axis: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (ranges, steps) = parse_selection(selection, &self.0.metadata().shape)?;
        if steps.iter().any(|step| *step != 1) {
            return Err(PyValueError::new_err("mean requires step-1 slices"));
        }
        let result = py
            .detach(|| self.0.read_mean(&ranges, axis, block))
            .map_err(python_error)?;
        numpy_region(py, result)
    }
}

fn open_remote(url: &str, pool: Option<ReadPool>, options: S3Options) -> Result<H5File, Error> {
    match pool {
        Some(pool) => H5File::open_s3_with_options(url, Limits::default(), pool, options),
        None => H5File::open_s3_with_defaults(url, options),
    }
}

#[pyfunction]
#[pyo3(signature = (urls, pool=None, *, prefetch_header=false, preload_chunk_index=false, client=None))]
fn open_s3_many(
    py: Python<'_>,
    urls: Vec<String>,
    pool: Option<PyRef<'_, PythonPool>>,
    prefetch_header: bool,
    preload_chunk_index: bool,
    client: Option<PyRef<'_, PythonS3Client>>,
) -> PyResult<Vec<PythonFile>> {
    let pool = pool.map(|pool| pool.0.clone());
    let mut options = S3Options::default();
    options.prefetch_header = prefetch_header;
    options.preload_chunk_index = preload_chunk_index;
    options.client = client.map(|client| client.0.clone());
    py.detach(|| {
        urls.par_iter()
            .map(|url| open_remote(url, pool.clone(), options.clone()).map(PythonFile))
            .collect::<Vec<_>>()
    })
    .into_iter()
    .map(|file| file.map_err(python_error))
    .collect()
}

type BatchRequest = (Py<PythonDataset>, Py<PyAny>);
type ParsedSelection = (Vec<Range<u64>>, Vec<isize>);

/// Read independent datasets and selections concurrently, in input order.
#[pyfunction]
fn read_many<'py>(
    py: Python<'py>,
    requests: Vec<BatchRequest>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let requests = requests
        .into_iter()
        .map(|(dataset, selection)| {
            let dataset = dataset.borrow(py).0.clone();
            let parsed = parse_selection(selection.bind(py), &dataset.metadata().shape)?;
            Ok((dataset, parsed))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let results = py.detach(|| {
        // Await each pool's admission permit without blocking a Rayon worker.
        pollster::block_on(futures_util::future::join_all(requests.iter().map(
            |(dataset, (ranges, steps))| dataset.read_strided_async(ranges, steps),
        )))
    });
    results
        .into_iter()
        .map(|result| numpy_region(py, result.map_err(python_error)?))
        .collect()
}

fn numpy_region(py: Python<'_>, result: Region) -> PyResult<Bound<'_, PyAny>> {
    result
        .bytes
        .into_pyarray(py)
        .call_method1("view", (numpy_type(result.dtype),))?
        .call_method1("reshape", (PyTuple::new(py, result.shape)?,))
}

fn parse_selection(selection: &Bound<'_, PyAny>, extents: &[u64]) -> PyResult<ParsedSelection> {
    if let Ok(tuple) = selection.cast::<PyTuple>() {
        if tuple.len() == extents.len() {
            return tuple
                .iter()
                .zip(extents)
                .map(|(slice, &extent)| slice_range(&slice, extent))
                .collect();
        }
    } else if extents.len() == 1 {
        let (range, step) = slice_range(selection, extents[0])?;
        return Ok((vec![range], vec![step]));
    }
    Err(PyIndexError::new_err("expected one slice per dataset axis"))
}

fn slice_range(slice: &Bound<'_, PyAny>, extent: u64) -> PyResult<(Range<u64>, isize)> {
    let slice = slice.cast::<PySlice>()?;
    let extent = isize::try_from(extent).map_err(|_| PyIndexError::new_err("axis too long"))?;
    let indices = slice.indices(extent)?;
    if indices.slicelength == 0 {
        return Ok((0..0, indices.step));
    }
    let last = indices.start + indices.step * (indices.slicelength as isize - 1);
    let range = indices.start.min(last) as u64..indices.start.max(last) as u64 + 1;
    Ok((range, indices.step))
}

fn numpy_type(dtype: DType) -> String {
    use NumericType::*;
    let order = match dtype.byte_order {
        ByteOrder::Little => '<',
        ByteOrder::Big => '>',
        ByteOrder::NotApplicable => '|',
    };
    let kind = match dtype.kind {
        I8 | I16 | I32 | I64 => 'i',
        U8 | U16 | U32 | U64 => 'u',
        F32 | F64 => 'f',
    };
    format!("{order}{kind}{}", dtype.size())
}

fn python_error(error: Error) -> PyErr {
    let message = error.to_string();
    let (kind, status, reason) = match &error {
        Error::CaBundle { reason, .. } => ("ca_bundle", None, *reason),
        Error::RemoteDns => ("dns", None, "host lookup failed"),
        Error::RemoteTls(reason) => ("tls", None, *reason),
        Error::RemoteConnect(reason) => ("connect", None, *reason),
        Error::RemoteTimeout(phase) => ("timeout", None, *phase),
        Error::RemoteStatus(status) => ("http", Some(*status), "unexpected HTTP status"),
        Error::InvalidRangeResponse(reason) => ("range_response", None, *reason),
        Error::ObjectChanged => (
            "object_changed",
            None,
            "object no longer matches its pinned identity",
        ),
        Error::RemoteTransport => ("transport", None, "HTTP transport failed"),
        Error::UrlRefresh => ("url_refresh", None, "could not refresh the signed URL"),
        _ => {
            return match error {
                Error::Unsupported(_) | Error::UnsupportedFilter(_) => {
                    PyNotImplementedError::new_err(message)
                }
                Error::SelectionRank { .. } | Error::SelectionBounds { .. } => {
                    PyIndexError::new_err(message)
                }
                Error::InvalidName
                | Error::InvalidConcurrency
                | Error::InvalidTransform
                | Error::InvalidS3Timeout
                | Error::InvalidRangeOptions => PyValueError::new_err(message),
                _ => PyIOError::new_err(message),
            }
        }
    };
    let error = S3Error::new_err(message);
    let _ = Python::attach(|py| {
        let value = error.value(py);
        value.setattr("kind", kind)?;
        value.setattr("status_code", status)?;
        value.setattr("reason", reason)
    });
    error
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonPool>()?;
    module.add_class::<PythonFile>()?;
    module.add_class::<PythonDataset>()?;
    module.add_class::<PythonS3Client>()?;
    module.add("S3Error", module.py().get_type::<S3Error>())?;
    module.add_function(wrap_pyfunction!(open_s3_many, module)?)?;
    module.add_function(wrap_pyfunction!(read_many, module)?)?;
    Ok(())
}
