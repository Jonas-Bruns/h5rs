use std::fs::File;
use std::io;
use std::path::Path;

use crate::{metadata::ChunkLocation, Error};

pub(crate) enum RawFile {
    Local(LocalFile),
    #[cfg(feature = "s3")]
    Remote(std::sync::Arc<crate::s3::Object>),
}

pub(crate) struct LocalFile {
    file: File,
    len: u64,
}

impl RawFile {
    pub(crate) fn open(path: &Path) -> Result<Self, Error> {
        #[cfg(unix)]
        let file = File::open(path)?;
        #[cfg(windows)]
        let file = windows::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(Error::Unsupported("non-regular file"));
        }
        Ok(Self::Local(LocalFile {
            file,
            len: metadata.len(),
        }))
    }

    pub(crate) fn validate_chunk(
        &self,
        location: &ChunkLocation,
        limit: usize,
    ) -> Result<usize, Error> {
        if location
            .offset
            .checked_add(location.size)
            .is_none_or(|end| end > self.len())
        {
            return Err(Error::InvalidChunk("stored bytes fall outside the file"));
        }
        let size = usize::try_from(location.size).map_err(|_| Error::Overflow("stored chunk"))?;
        if size > limit {
            return Err(Error::Limit {
                kind: "stored chunk",
                bytes: size,
                limit,
            });
        }
        Ok(size)
    }

    fn len(&self) -> u64 {
        match self {
            Self::Local(local) => local.len,
            #[cfg(feature = "s3")]
            Self::Remote(object) => object.len(),
        }
    }

    pub(crate) fn read_exact_at(&self, buffer: &mut [u8], offset: u64) -> Result<(), Error> {
        match self {
            Self::Local(local) => Ok(local.read_exact_at(buffer, offset)?),
            #[cfg(feature = "s3")]
            Self::Remote(object) => object.read_exact_at(buffer, offset),
        }
    }
}

impl LocalFile {
    #[cfg(unix)]
    fn read_exact_at(&self, buffer: &mut [u8], offset: u64) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buffer, offset)
    }

    #[cfg(windows)]
    fn read_exact_at(&self, buffer: &mut [u8], offset: u64) -> io::Result<()> {
        windows::read_exact_at(&self.file, buffer, offset)
    }
}

#[cfg(windows)]
mod windows {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use std::path::Path;
    use std::ptr;

    use windows_sys::Win32::Foundation::{ERROR_HANDLE_EOF, ERROR_IO_PENDING};
    use windows_sys::Win32::Storage::FileSystem::{ReadFile, FILE_FLAG_OVERLAPPED};
    use windows_sys::Win32::System::Threading::CreateEventW;
    use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED, OVERLAPPED_0_0};

    pub(super) fn open(path: &Path) -> io::Result<File> {
        OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_OVERLAPPED)
            .open(path)
    }

    pub(super) fn read_exact_at(
        file: &File,
        mut buffer: &mut [u8],
        mut offset: u64,
    ) -> io::Result<()> {
        // SAFETY: Null attributes and name request an unnamed, non-inheritable
        // event. A successful call transfers ownership of the handle to us.
        let event = unsafe { CreateEventW(ptr::null(), 1, 0, ptr::null()) };
        if event.is_null() {
            return Err(io::Error::last_os_error());
        }
        let event = unsafe { OwnedHandle::from_raw_handle(event) };
        while !buffer.is_empty() {
            let mut operation = OVERLAPPED {
                hEvent: event.as_raw_handle(),
                ..OVERLAPPED::default()
            };
            operation.Anonymous.Anonymous = OVERLAPPED_0_0 {
                Offset: offset as u32,
                OffsetHigh: (offset >> 32) as u32,
            };
            let file = file.as_raw_handle();
            let count = buffer.len().min(u32::MAX as usize) as u32;
            // SAFETY: This file was opened for overlapped I/O. The operation,
            // event and writable buffer stay live until GetOverlappedResult
            // observes completion. Every outstanding read has its own event.
            let started = unsafe {
                ReadFile(
                    file,
                    buffer.as_mut_ptr(),
                    count,
                    ptr::null_mut(),
                    &mut operation,
                )
            };
            if started == 0 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(ERROR_IO_PENDING as i32) {
                    return Err(as_read_error(error));
                }
            }
            let mut transferred = 0;
            if unsafe { GetOverlappedResult(file, &operation, &mut transferred, 1) } == 0 {
                return Err(as_read_error(io::Error::last_os_error()));
            }
            if transferred == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            buffer = &mut buffer[transferred as usize..];
            offset += u64::from(transferred);
        }
        Ok(())
    }

    fn as_read_error(error: io::Error) -> io::Error {
        if error.raw_os_error() == Some(ERROR_HANDLE_EOF as i32) {
            io::ErrorKind::UnexpectedEof.into()
        } else {
            error
        }
    }
}
