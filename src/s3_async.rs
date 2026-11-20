use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use http_body_util::{BodyExt, Empty};
use hyper::body::{Bytes, Incoming};
use hyper::{header, Request, Response};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::{TokioExecutor, TokioTimer};
use tokio::runtime::{Builder, Runtime};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use ureq::http::Uri;

use crate::s3::{validate_range_headers, Object};
use crate::{error::zeroed, Error};

pub(crate) struct Transport {
    pub limit: usize,
    permits: Semaphore,
    client: Client<HttpsConnector<HttpConnector>, Empty<Bytes>>,
    runtime: &'static Runtime,
}

impl Transport {
    pub(crate) fn new(requests: usize) -> Result<Arc<Self>, Error> {
        static RUNTIME: OnceLock<Result<Runtime, std::io::Error>> = OnceLock::new();
        if requests == 0 || requests > Semaphore::MAX_PERMITS {
            return Err(Error::InvalidConcurrency);
        }
        let runtime = RUNTIME
            .get_or_init(|| {
                Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
            })
            .as_ref()
            .map_err(|_| Error::RemoteTransport)?;
        let connector = HttpsConnectorBuilder::new()
            .with_webpki_roots()
            .https_or_http()
            .enable_http1()
            .build();
        let client = Client::builder(TokioExecutor::new())
            .pool_max_idle_per_host(requests)
            .pool_timer(TokioTimer::new())
            .build(connector);
        Ok(Arc::new(Self {
            limit: requests,
            permits: Semaphore::new(requests),
            client,
            runtime,
        }))
    }

    pub(crate) fn read(
        self: &Arc<Self>,
        object: &Arc<Object>,
        ranges: Vec<(u64, usize)>,
    ) -> Result<Vec<Vec<u8>>, Error> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        let transport = Arc::clone(self);
        let object = Arc::clone(object);
        self.runtime.spawn(async move {
            let result = transport.read_batch(object, ranges).await;
            let _ = send.send(result);
        });
        receive.recv().map_err(|_| Error::WorkerTerminated)?
    }

    async fn read_batch(
        self: Arc<Self>,
        object: Arc<Object>,
        ranges: Vec<(u64, usize)>,
    ) -> Result<Vec<Vec<u8>>, Error> {
        let mut output = vec![Vec::new(); ranges.len()];
        let mut pending = ranges.into_iter().enumerate();
        let mut requests = JoinSet::new();
        loop {
            while requests.len() < self.limit {
                let Some((index, (offset, size))) = pending.next() else {
                    break;
                };
                let transport = Arc::clone(&self);
                let object = Arc::clone(&object);
                requests.spawn(async move {
                    let _permit = transport
                        .permits
                        .acquire()
                        .await
                        .map_err(|_| Error::WorkerTerminated)?;
                    tokio::time::timeout(
                        Duration::from_secs(30),
                        transport.get(&object, offset, size),
                    )
                    .await
                    .map_err(|_| Error::RemoteTransport)?
                    .map(|bytes| (index, bytes))
                });
            }
            let Some(result) = requests.join_next().await else {
                return Ok(output);
            };
            let (index, bytes) = result.map_err(|_| Error::WorkerTerminated)??;
            output[index] = bytes;
        }
    }

    async fn get(&self, object: &Arc<Object>, offset: u64, size: usize) -> Result<Vec<u8>, Error> {
        let end = offset
            .checked_add(size as u64)
            .filter(|end| *end <= object.len() && *end > offset)
            .ok_or(Error::InvalidRangeResponse(
                "requested range outside object",
            ))?;
        let range = offset..end;
        let url = object.current_url();
        let mut response = match self.request(object, &url, &range).await {
            Err(Error::RemoteStatus(401 | 403)) if object.can_refresh() => {
                let refresh_object = Arc::clone(object);
                let url = tokio::task::spawn_blocking(move || refresh_object.renew_url(&url))
                    .await
                    .map_err(|_| Error::WorkerTerminated)??;
                // The replacement identity is validated before reading any bytes.
                self.request(object, &url, &range).await?
            }
            other => other?,
        };
        let mut output = zeroed(size)?;
        let mut remaining = output.as_mut_slice();
        while let Some(frame) = response.body_mut().frame().await {
            let frame = frame
                .map_err(|_| Error::InvalidRangeResponse("incomplete or unreadable HTTP body"))?;
            let Ok(bytes) = frame.into_data() else {
                continue;
            };
            if bytes.len() > remaining.len() {
                return Err(Error::InvalidRangeResponse("unexpected body length"));
            }
            let (target, tail) = remaining.split_at_mut(bytes.len());
            target.copy_from_slice(&bytes);
            remaining = tail;
        }
        if !remaining.is_empty() {
            return Err(Error::InvalidRangeResponse("unexpected body length"));
        }
        Ok(output)
    }

    async fn request(
        &self,
        object: &Object,
        url: &Uri,
        range: &Range<u64>,
    ) -> Result<Response<Incoming>, Error> {
        let request = Request::get(url.clone())
            .header(
                header::RANGE,
                format!("bytes={}-{}", range.start, range.end - 1),
            )
            .header(header::IF_MATCH, object.etag().clone())
            .header(header::ACCEPT_ENCODING, "identity")
            .body(Empty::new())
            .map_err(|_| Error::RemoteTransport)?;
        let response = self
            .client
            .request(request)
            .await
            .map_err(|_| Error::RemoteTransport)?;
        match response.status().as_u16() {
            206 => {}
            200 => {
                return Err(Error::InvalidRangeResponse(
                    "server ignored the Range header",
                ))
            }
            412 => return Err(Error::ObjectChanged),
            status => return Err(Error::RemoteStatus(status)),
        }
        validate_range_headers(response.headers(), range.clone(), Some(object.len()))?;
        if response.headers().get(header::ETAG) != Some(object.etag()) {
            return Err(Error::ObjectChanged);
        }
        Ok(response)
    }
}
