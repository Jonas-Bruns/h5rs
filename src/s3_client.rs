use std::error::Error as StdError;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use rustls::pki_types::{pem::PemObject, CertificateDer};
use ureq::config::Config;
use ureq::http::Uri;
use ureq::tls::{Certificate, RootCerts, TlsConfig};
use ureq::unversioned::resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver};
use ureq::unversioned::transport::{DefaultConnector, NextTimeout};
use ureq::Agent;

use crate::Error;

/// TLS trust and timeout configuration shared by remote objects.
#[derive(Clone)]
pub struct S3ClientConfig {
    /// A PEM trust bundle. Otherwise use SSL_CERT_FILE or AWS_CA_BUNDLE.
    pub ca_bundle: Option<PathBuf>,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
}

impl Default for S3ClientConfig {
    fn default() -> Self {
        Self {
            ca_bundle: None,
            connect_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(30),
        }
    }
}

/// A reusable set of HTTP connection pools. Object identity stays per file.
#[derive(Clone)]
pub struct S3Client(Arc<ClientInner>);

struct ClientInner {
    agent: Agent,
    #[cfg(feature = "s3-async-io")]
    asynchronous: crate::s3_async::HttpClient,
    #[cfg(feature = "s3-async-io")]
    request_timeout: Duration,
}

#[derive(Debug, Default)]
struct DnsResolver(DefaultResolver);

impl Resolver for DnsResolver {
    fn resolve(
        &self,
        uri: &Uri,
        config: &Config,
        timeout: NextTimeout,
    ) -> Result<ResolvedSocketAddrs, ureq::Error> {
        self.0
            .resolve(uri, config, timeout)
            .map_err(|error| match error {
                ureq::Error::Io(source) if source.kind() == io::ErrorKind::TimedOut => {
                    ureq::Error::Timeout(ureq::Timeout::Resolve)
                }
                ureq::Error::Io(_) => ureq::Error::HostNotFound,
                other => other,
            })
    }
}

impl S3Client {
    /// Create independent connection pools with the selected trust settings.
    pub fn new(mut config: S3ClientConfig) -> Result<Self, Error> {
        config.ca_bundle = config.ca_bundle.or_else(environment_bundle);
        Self::build(config)
    }

    pub(crate) fn shared() -> Result<Self, Error> {
        static DEFAULT: Mutex<Option<(Option<PathBuf>, S3Client)>> = Mutex::new(None);
        let ca_bundle = environment_bundle();
        let mut default = DEFAULT.lock();
        if let Some((path, client)) = &*default {
            if *path == ca_bundle {
                return Ok(client.clone());
            }
        }
        let client = Self::build(S3ClientConfig {
            ca_bundle: ca_bundle.clone(),
            ..S3ClientConfig::default()
        })?;
        *default = Some((ca_bundle, client.clone()));
        Ok(client)
    }

    fn build(config: S3ClientConfig) -> Result<Self, Error> {
        for timeout in [config.connect_timeout, config.request_timeout] {
            if timeout.is_zero() || std::time::Instant::now().checked_add(timeout).is_none() {
                return Err(Error::InvalidS3Timeout);
            }
        }
        let bundle = config.ca_bundle.as_deref().map(load_bundle).transpose()?;
        let roots = match &bundle {
            Some((certificates, _)) => certificates
                .iter()
                .map(|cert| Certificate::from_der(cert.as_ref()).to_owned())
                .collect::<Vec<_>>()
                .into(),
            None => RootCerts::WebPki,
        };
        let agent_config = Agent::config_builder()
            .tls_config(TlsConfig::builder().root_certs(roots).build())
            .timeout_global(Some(config.request_timeout))
            .timeout_connect(Some(config.connect_timeout))
            .max_redirects(0)
            .max_redirects_will_error(false)
            .max_idle_connections(256)
            .max_idle_connections_per_host(64)
            .build();
        let agent = Agent::with_parts(
            agent_config,
            DefaultConnector::default(),
            DnsResolver::default(),
        );
        Ok(Self(Arc::new(ClientInner {
            agent,
            #[cfg(feature = "s3-async-io")]
            asynchronous: crate::s3_async::http_client(
                bundle.map(|(_, roots)| roots),
                config.connect_timeout,
            ),
            #[cfg(feature = "s3-async-io")]
            request_timeout: config.request_timeout,
        })))
    }

    pub(crate) fn agent(&self) -> &Agent {
        &self.0.agent
    }

    #[cfg(feature = "s3-async-io")]
    pub(crate) fn asynchronous(&self) -> &crate::s3_async::HttpClient {
        &self.0.asynchronous
    }

    #[cfg(feature = "s3-async-io")]
    pub(crate) fn request_timeout(&self) -> Duration {
        self.0.request_timeout
    }
}

fn environment_bundle() -> Option<PathBuf> {
    ["SSL_CERT_FILE", "AWS_CA_BUNDLE"]
        .into_iter()
        .find_map(|name| std::env::var_os(name).filter(|value| !value.is_empty()))
        .map(PathBuf::from)
}

fn load_bundle(
    path: &Path,
) -> Result<(Vec<CertificateDer<'static>>, rustls::RootCertStore), Error> {
    let error = |reason| Error::CaBundle {
        path: path.to_owned(),
        reason,
    };
    let bytes = std::fs::read(path).map_err(|_| error("could not read the PEM file"))?;
    let certificates = CertificateDer::pem_slice_iter(&bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| error("could not parse the PEM certificates"))?;
    if certificates.is_empty() {
        return Err(error("contains no PEM certificates"));
    }
    let mut roots = rustls::RootCertStore::empty();
    for cert in &certificates {
        roots
            .add(cert.clone())
            .map_err(|_| error("contains an invalid X.509 certificate"))?;
    }
    Ok((certificates, roots))
}

pub(crate) fn tls_error(error: &rustls::Error) -> Error {
    use rustls::CertificateError;
    let reason = match error {
        rustls::Error::InvalidCertificate(CertificateError::UnknownIssuer) => {
            "certificate issuer is not trusted"
        }
        rustls::Error::InvalidCertificate(
            CertificateError::Expired | CertificateError::ExpiredContext { .. },
        ) => "certificate has expired",
        rustls::Error::InvalidCertificate(
            CertificateError::NotValidYet | CertificateError::NotValidYetContext { .. },
        ) => "certificate is not valid yet",
        rustls::Error::InvalidCertificate(
            CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. },
        ) => "certificate hostname does not match",
        rustls::Error::InvalidCertificate(_) => "certificate verification failed",
        _ => "TLS handshake or protocol failed",
    };
    Error::RemoteTls(reason)
}

pub(crate) fn transport_error(error: &(dyn StdError + 'static)) -> Error {
    let mut current = Some(error);
    let mut io_kind = None;
    while let Some(cause) = current {
        if let Some(error) = cause.downcast_ref::<ureq::Error>() {
            match error {
                ureq::Error::HostNotFound => return Error::RemoteDns,
                ureq::Error::Timeout(phase) => {
                    return Error::RemoteTimeout(match phase {
                        ureq::Timeout::Resolve => "DNS resolution",
                        ureq::Timeout::Connect => "connection establishment",
                        _ => "HTTP request",
                    });
                }
                ureq::Error::Tls(_) | ureq::Error::TlsRequired => {
                    return Error::RemoteTls("TLS handshake or protocol failed")
                }
                ureq::Error::Rustls(error) => return tls_error(error),
                ureq::Error::ConnectionFailed => {
                    return Error::RemoteConnect("connection could not be established")
                }
                ureq::Error::ConnectProxyFailed(_) | ureq::Error::InvalidProxyUrl => {
                    return Error::RemoteConnect("proxy connection failed")
                }
                ureq::Error::Io(error) => {
                    current = Some(error);
                    continue;
                }
                _ => {}
            }
        }
        if let Some(error) = cause.downcast_ref::<rustls::Error>() {
            return tls_error(error);
        }
        if let Some(error) = cause.downcast_ref::<io::Error>() {
            io_kind = Some(error.kind());
            current = error
                .get_ref()
                .map(|inner| inner as &dyn StdError)
                .or_else(|| cause.source());
            continue;
        }
        #[cfg(feature = "s3-async-io")]
        if let Some(error) = cause.downcast_ref::<crate::s3_async::ConnectFailure>() {
            return error.as_remote();
        }
        current = cause.source();
    }
    match io_kind {
        Some(io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock) => {
            Error::RemoteTimeout("HTTP request")
        }
        Some(io::ErrorKind::ConnectionRefused) => Error::RemoteConnect("connection refused"),
        Some(io::ErrorKind::ConnectionReset) => Error::RemoteConnect("connection reset"),
        Some(io::ErrorKind::ConnectionAborted) => Error::RemoteConnect("connection aborted"),
        Some(
            io::ErrorKind::NotConnected
            | io::ErrorKind::NetworkUnreachable
            | io::ErrorKind::HostUnreachable,
        ) => Error::RemoteConnect("endpoint is unreachable"),
        _ => Error::RemoteTransport,
    }
}
