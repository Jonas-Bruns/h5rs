use std::ops::Range;
use std::path::PathBuf;

use h5rs_core::{ByteOrder, DType, Error, H5File, Limits, NumericType, ReadPool};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyIOError, PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PySliceMethods, PyTuple};

#[pyclass(name = "ReadPool", module = "h5rs", frozen)]
struct PythonPool(ReadPool);

#[pymethods]
impl PythonPool {
    #[new]
    fn new(threads: usize, concurrent_requests: usize) -> PyResult<Self> {
        Ok(Self(
            ReadPool::new(threads, concurrent_requests).map_err(python_error)?,
        ))
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
    #[pyo3(signature = (url, pool=None))]
    fn open_s3(py: Python<'_>, url: &str, pool: Option<PyRef<'_, PythonPool>>) -> PyResult<Self> {
        let pool = pool.map(|pool| pool.0.clone());
        py.detach(move || match pool {
            Some(pool) => H5File::open_s3_with_pool(url, Limits::default(), pool),
            None => H5File::open_s3(url),
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
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.metadata().shape.iter().copied())
    }

    #[getter]
    fn chunks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.metadata().chunks.iter().copied())
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
        selection: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let extents = &self.0.metadata().shape;
        if selection.len() != extents.len() {
            return Err(PyIndexError::new_err("expected one slice per dataset axis"));
        }
        let ranges = selection
            .iter()
            .zip(extents)
            .map(|(slice, &extent)| slice_range(&slice, extent))
            .collect::<PyResult<Vec<_>>>()?;
        let result = py
            .detach(|| self.0.read_region(&ranges))
            .map_err(python_error)?;
        // NumPy owns a view of the Rust allocation. The dtype can retain the
        // file's byte order, and view/reshape do not copy the buffer.
        result
            .bytes
            .into_pyarray(py)
            .call_method1("view", (numpy_type(result.dtype),))?
            .call_method1("reshape", (PyTuple::new(py, result.shape)?,))
    }
}

fn slice_range(slice: &Bound<'_, PyAny>, extent: u64) -> PyResult<Range<u64>> {
    let slice = slice.cast::<PySlice>()?;
    let extent = isize::try_from(extent).map_err(|_| PyIndexError::new_err("axis too long"))?;
    let indices = slice.indices(extent)?;
    if indices.step != 1 {
        return Err(PyNotImplementedError::new_err(
            "only step-1 slices are supported",
        ));
    }
    Ok(indices.start as u64..indices.stop.max(indices.start) as u64)
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
    match error {
        Error::Unsupported(_) | Error::UnsupportedFilter(_) => {
            PyNotImplementedError::new_err(message)
        }
        Error::SelectionRank { .. } | Error::SelectionBounds { .. } => {
            PyIndexError::new_err(message)
        }
        Error::InvalidName | Error::InvalidConcurrency => PyValueError::new_err(message),
        _ => PyIOError::new_err(message),
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonPool>()?;
    module.add_class::<PythonFile>()?;
    module.add_class::<PythonDataset>()?;
    Ok(())
}
