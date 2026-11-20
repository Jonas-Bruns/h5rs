use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use http_body_util::{BodyExt, Empty};
use hyper::body::{Bytes, Incoming};
use hyper::{header, Request, Response};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::connect::dns::{GaiAddrs, GaiResolver, Name};
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::{TokioExecutor, TokioTimer};
use tokio::runtime::{Builder, Runtime};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tower_service::Service;
use ureq::http::Uri;

use crate::s3::{validate_range_headers, Object};
use crate::s3_client::transport_error;
use crate::{error::zeroed, Error};

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Connection = <HttpsConnector<HttpConnector<Dns>> as Service<Uri>>::Response;
pub(crate) type HttpClient = Client<Connector, Empty<Bytes>>;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ConnectFailure {
    #[error("DNS resolution failed")]
    Dns,
    #[error("connection establishment timed out")]
    Timeout,
}

impl ConnectFailure {
    pub(crate) fn as_remote(&self) -> Error {
        match self {
            Self::Dns => Error::RemoteDns,
            Self::Timeout => Error::RemoteTimeout("connection establishment"),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Dns(GaiResolver);

impl Service<Name> for Dns {
    type Response = GaiAddrs;
    type Error = ConnectFailure;
    type Future = Pin<Box<dyn Future<Output = Result<GaiAddrs, ConnectFailure>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.0.poll_ready(cx).map_err(|_| ConnectFailure::Dns)
    }

    fn call(&mut self, name: Name) -> Self::Future {
        let resolve = self.0.call(name);
        Box::pin(async move { resolve.await.map_err(|_| ConnectFailure::Dns) })
    }
}

#[derive(Clone)]
pub(crate) struct Connector {
    inner: HttpsConnector<HttpConnector<Dns>>,
    timeout: Duration,
}

impl Service<Uri> for Connector {
    type Response = Connection;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Connection, BoxError>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let connection = self.inner.call(uri);
        let timeout = self.timeout;
        Box::pin(async move {
            tokio::time::timeout(timeout, connection)
                .await
                .map_err(|_| Box::new(ConnectFailure::Timeout) as BoxError)?
        })
    }
}

pub(crate) fn http_client(roots: Option<rustls::RootCertStore>, timeout: Duration) -> HttpClient {
    let builder = HttpsConnectorBuilder::new();
    let builder = match roots {
        Some(roots) => builder.with_tls_config(
            rustls::ClientConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .expect("ring supported protocols")
            .with_root_certificates(roots)
            .with_no_client_auth(),
        ),
        None => builder.with_webpki_roots(),
    };
    let mut tcp = HttpConnector::new_with_resolver(Dns(GaiResolver::new()));
    tcp.enforce_http(false);
    let connector = Connector {
        inner: builder.https_or_http().enable_http1().wrap_connector(tcp),
        timeout,
    };
    Client::builder(TokioExecutor::new())
        .pool_max_idle_per_host(64)
        .pool_timer(TokioTimer::new())
        .build(connector)
}

pub(crate) struct Transport {
    pub limit: usize,
    permits: Semaphore,
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
        Ok(Arc::new(Self {
            limit: requests,
            permits: Semaphore::new(requests),
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
                    transport
                        .get(&object, offset, size)
                        .await
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
        match self.fetch(object, &url, &range, size).await {
            Err(Error::RemoteStatus(401 | 403)) if object.can_refresh() => {
                let refresh_object = Arc::clone(object);
                let url = tokio::task::spawn_blocking(move || refresh_object.renew_url(&url))
                    .await
                    .map_err(|_| Error::WorkerTerminated)??;
                // The replacement identity is validated before reading any bytes.
                self.fetch(object, &url, &range, size).await
            }
            other => other,
        }
    }

    async fn fetch(
        &self,
        object: &Object,
        url: &Uri,
        range: &Range<u64>,
        size: usize,
    ) -> Result<Vec<u8>, Error> {
        tokio::time::timeout(
            object.client.request_timeout(),
            self.fetch_body(object, url, range, size),
        )
        .await
        .map_err(|_| Error::RemoteTimeout("HTTP request"))?
    }

    async fn fetch_body(
        &self,
        object: &Object,
        url: &Uri,
        range: &Range<u64>,
        size: usize,
    ) -> Result<Vec<u8>, Error> {
        let mut response = self.request(object, url, range).await?;
        let mut output = zeroed(size)?;
        let mut remaining = output.as_mut_slice();
        while let Some(frame) = response.body_mut().frame().await {
            let frame = frame.map_err(|error| match transport_error(&error) {
                Error::RemoteTransport => {
                    Error::InvalidRangeResponse("incomplete or unreadable HTTP body")
                }
                classified => classified,
            })?;
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
        let response = object
            .client
            .asynchronous()
            .request(request)
            .await
            .map_err(|error| match transport_error(&error) {
                Error::RemoteTransport if error.is_connect() => {
                    Error::RemoteConnect("connection could not be established")
                }
                classified => classified,
            })?;
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
