use std::sync::{Arc, OnceLock};

use async_lock::Semaphore;
use futures_channel::oneshot;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::Error;

/// A worker pool and admission limit shared by one or more HDF5 files.
#[derive(Clone)]
pub struct ReadPool(Arc<Inner>);

struct Inner {
    workers: ThreadPool,
    permits: Arc<Semaphore>,
}

impl ReadPool {
    /// Allow at most `concurrent_requests` requests to allocate scratch and
    /// produce results at once, using at most `threads` worker threads.
    pub fn new(threads: usize, concurrent_requests: usize) -> Result<Self, Error> {
        if threads == 0 || threads > rayon::max_num_threads() || concurrent_requests == 0 {
            return Err(Error::InvalidConcurrency);
        }
        let workers = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("h5rs-{index}"))
            .build()?;
        Ok(Self(Arc::new(Inner {
            workers,
            permits: Arc::new(Semaphore::new(concurrent_requests)),
        })))
    }

    pub(crate) fn shared() -> Result<Self, Error> {
        static POOL: OnceLock<ReadPool> = OnceLock::new();
        static INIT: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
        if let Some(pool) = POOL.get() {
            return Ok(pool.clone());
        }
        let _guard = INIT.lock();
        if let Some(pool) = POOL.get() {
            return Ok(pool.clone());
        }
        let threads = std::thread::available_parallelism().map_or(1, |count| count.get().min(8));
        let pool = Self::new(threads, threads)?;
        let _ = POOL.set(pool.clone());
        Ok(pool)
    }

    pub(crate) fn threads(&self) -> usize {
        self.0.workers.current_num_threads()
    }

    pub(crate) fn run<T: Send>(
        &self,
        read: impl FnOnce() -> Result<T, Error> + Send,
    ) -> Result<T, Error> {
        let _permit = self.0.permits.acquire_blocking();
        self.0.workers.install(read)
    }

    pub(crate) async fn submit<T: Send + 'static>(
        &self,
        read: impl FnOnce() -> Result<T, Error> + Send + 'static,
    ) -> Result<T, Error> {
        let permit = self.0.permits.acquire_arc().await;
        let (send, receive) = oneshot::channel();
        // Keep the slot with the worker until completion, then with any result
        // still queued for the caller. Dropping the future releases a queued
        // result, while already-running work keeps its slot.
        self.0.workers.spawn(move || {
            if !send.is_canceled() {
                // Rayon aborts on unhandled panics in detached jobs. The waiter
                // receives WorkerTerminated if an unexpected panic unwinds here.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(read));
                let _ = send.send((result.unwrap_or(Err(Error::WorkerTerminated)), permit));
            }
        });
        let (result, _permit) = receive.await.map_err(|_| Error::WorkerTerminated)?;
        result
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::mpsc;
    use std::task::{Context, Poll, Waker};

    use super::*;

    #[test]
    fn async_wait_yields_and_canceled_work_holds_its_permit_until_done() {
        let pool = ReadPool::new(1, 1).unwrap();
        let (started, running) = mpsc::channel();
        let (release, blocked) = mpsc::channel();
        let mut first = Box::pin(pool.submit(move || {
            started.send(()).unwrap();
            blocked.recv().unwrap();
            Ok(())
        }));
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(first.as_mut().poll(&mut context), Poll::Pending));
        running
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        drop(first);

        let mut second = Box::pin(pool.submit(|| Ok(42)));
        assert!(matches!(second.as_mut().poll(&mut context), Poll::Pending));
        release.send(()).unwrap();
        assert_eq!(pollster::block_on(second).unwrap(), 42);
    }
}
