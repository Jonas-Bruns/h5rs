# h5rs

`h5rs` reads rectangular regions of numeric HDF5 datasets in immutable local
files and S3 objects. The bundled HDF5 C library handles metadata and chunk
locations. Rust reads stored chunks and contiguous spans by offset. It reverses
chunk filters and copies the intersecting rows into the output.

## Rust API

```rust
use h5rs::H5File;

fn main() -> Result<(), h5rs::Error> {
    let file = H5File::open("data.h5")?;
    let dataset = file.dataset("/measurements/channel_1")?;
    let tile = dataset.read_region(&[100..356, 200..456])?;
    assert_eq!(tile.shape, [256, 256]);
    println!("{:?}, {} bytes", tile.dtype, tile.bytes.len());
    Ok(())
}
```

`Dataset::metadata` returns the shape, chunk shape, numeric type, byte order,
filter pipeline and fill value. `read_region` returns packed bytes in row-major
order with the original byte order. For example, an `I16` element with
`ByteOrder::Big` can be decoded with `i16::from_be_bytes`. Use
`read_region_into` to reuse an exact-length output buffer. On a read error this
buffer can contain partial results.

## Concurrency

Both file and dataset handles can be cloned and shared across Rust threads.
The default pool spans all files and uses up to eight workers based on CPU
availability. Use `ReadPool::new(threads, concurrent_requests)` with
`H5File::open_with_pool` or `H5File::open_s3_with_pool` to share a custom
pool across files. A one-thread pool gives a sequential chunk baseline.

Chunk reads and decoding can overlap within a selection and across concurrent
selections. Unix uses positioned reads and Windows uses overlapped offset
reads. Metadata calls use a process-wide lock to support the bundled HDF5
library. A cold S3 metadata read may hold that lock while waiting for the
network.

On the first nonempty local read, h5rs attempts to build an in-memory index
of the dataset's allocated chunk addresses. Later reads can resolve those
addresses without acquiring the HDF5 lock. Indexes are limited to 65,536
entries per dataset handle and 64 MiB of reserved entry storage per
process. Local reads use HDF5 coordinate lookups if the index exceeds
these limits. For S3, prebuilding the in-memory index is opt-in.

For unfiltered local datasets, the reader combines up to four exactly
adjacent chunks into one read of at most 512 KiB. On S3 it can group up to
16 chunks and 2 MiB per request, including compressed chunks. Larger
individual chunks use separate reads. Independent spans can still be
read in parallel. A combined read covers only requested chunks.

Use `open_async`, `dataset_async` and `read_region_async` from any Rust async
runtime. The futures wait asynchronously for pool admission and results.
Workers perform blocking file or HTTP I/O. Dropping a future stops waiting,
and an operation that has already started can keep its worker and admission
slot until it finishes.

`H5File::open_with_limits` controls per-call memory limits. The defaults are
512 MiB of output and 64 MiB each for a stored or decoded chunk. Requests
plan batches by worker count and storage backend. They reserve at most
twice the chunk limit plus one byte for chunk buffers. Pool admission
bounds the number of simultaneous requests. Results retained by callers
and codec or transport state are outside these limits.

## S3

Enable the `s3` Cargo feature and pass a public object URL or a presigned
S3 HTTPS GET URL.

```rust
async fn tile(url: &str) -> Result<h5rs::Region, h5rs::Error> {
    let file = h5rs::H5File::open_s3_async(url).await?;
    let dataset = file.dataset_async("/measurements/channel_1").await?;
    dataset.read_region_async(&[100..356, 200..456]).await
}
```

Presign a GET whose signed headers allow `Range` and `If-Match` to vary.
The reader pins the initial strong ETag and object size, sends conditional
range requests and validates the returned byte ranges. A metadata-only HDF5
file driver caches up to 1 MiB of range pages per open object. Chunk GETs
use a shared HTTP connection pool per object. A versioned presigned URL
can select a specific S3 object version. URL signing and credential
discovery are handled by the caller.

The S3 integration test uses a local HTTP server and verifies overlapping
range GETs, byte-for-byte data, object-change detection and malformed
responses. Live AWS throughput and latency have not been measured.

### Optional S3 chunk address indexes in RAM

Prebuild an index directly from the read-only S3 object. This enumerates
allocated chunks in one HDF5 traversal and stores the addresses in memory.
Reuse the resulting dataset handle across requests.

```python
import h5rs

file = h5rs.File.open_s3(url)
dataset = file.dataset_with_s3_index("channel_0")
tile = dataset[10_000:14_864, :]
```

Rust callers use `H5File::dataset_with_s3_index`. The initial build can
fetch metadata from multiple S3 pages, so prebuild at service startup
for datasets you expect to read repeatedly. The index lives with the
dataset handle and uses the process-wide 64 MiB address-index budget.
Ordinary dataset access, such as `file["channel_0"]`, resolves addresses
on demand for one-off reads or datasets over the memory budget. The
S3 reader keeps checking the object's ETag on data reads.

## Compatibility

| Feature | Baseline support |
| --- | --- |
| File access | Read-only local files via native VOL and sec2, plus S3 HTTPS range reads with the `s3` feature. Local files and paths must remain stable while open. |
| Layout | Numeric 1D, 2D and 3D datasets. Chunked with default chunk options, contiguous in the HDF5 file, and compact. |
| Selections | One half-open range per axis. Strided reads accept positive or negative steps. Empty selections preserve their shape. |
| Types | Signed and unsigned 8-, 16-, 32- and 64-bit integers, plus IEEE float32 and float64, with little- and big-endian encodings. |
| Filters | No filters, Shuffle, Deflate (HDF5 gzip), or Shuffle followed by Deflate. Decoding honors each chunk's filter mask. |
| Missing chunks | Default and user-defined fill values with allocation or if-set fill timing. |
| Unsupported layouts and types | Contiguous, compact, virtual, compound, string and variable-length datasets return errors. External links are rejected. |

The baseline has been verified on Windows x64 with MSVC. CI is configured for
Linux, macOS and Windows. Build with stable Rust, CMake and a native C toolchain.
HDF5 is linked statically, so an installed HDF5 runtime is unnecessary.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
```

To include the h5py integration test, install the reference dependencies in a
Python environment on `PATH`, or set `H5RS_TEST_PYTHON` to its Python
executable.

```sh
python -m pip install "h5py>=3.16,<4"
cargo test --locked -- --include-ignored
cargo test --locked --all-features -- --include-ignored
```

The test generates small fixtures and compares output shape, datatype,
byte order and every byte against h5py across 94 selections. It also checks
concurrent requests, corrupt chunks, truncated files and unsupported features.

## Python

Install the published wheel from PyPI.

```sh
python -m pip install h5rs
```

The release builds wheels for CPython 3.10–3.14 on Windows x64, Linux x64
(glibc 2.28 or newer) and macOS Apple Silicon. Building from source requires
Rust, CMake and a native C toolchain. For local development, activate a
virtual environment and run the following from the repository root.

```sh
python -m pip install "maturin>=1.16,<2"
maturin develop --release --locked
```

```python
import h5rs

pool = h5rs.ReadPool(threads=8, concurrent_requests=8)
file = h5rs.File("distance_000.h5", pool)
tile = file["channel_0"][100:612, 0:512]
```

The Python reader accepts one slice per axis and returns a NumPy
view backed by the Rust output allocation. It releases the Python interpreter
while opening files, obtaining dataset metadata and reading regions.
To open a remote object, use `h5rs.File.open_s3(url, pool)` with a public
or presigned HTTPS GET URL for your S3-compatible endpoint. Use the same
dataset and slice API.

### S3 client settings and errors (since 0.2.1)

Reuse an S3 client across files that share a TLS and timeout configuration.
Python timeout values are in seconds. The request timeout applies to each
GET attempt, including its response body.

```python
client = h5rs.S3Client(ca_bundle="/etc/certs/minio-ca.pem",
                      connect_timeout=5.0, request_timeout=20.0)
pool = h5rs.ReadPool(threads=8, concurrent_requests=8, max_s3_gets=32)
files = h5rs.open_s3_many(urls, pool, client=client)
file = h5rs.File.open_s3(url, pool, client=client, refresh_url=sign_get)
```

The explicit PEM bundle takes precedence over `SSL_CERT_FILE`, then
`AWS_CA_BUNDLE`. A selected bundle supplies the trusted roots. The built-in
WebPKI roots are used when all three are unset. Configure CA and timeouts
directly on `File.open_s3` for a single file. Use a shared client for
repeated opens with custom settings. Default opens share a client for the
current environment trust settings. Synchronous and asynchronous HTTP
each maintain their own connection pool, and each object retains its
own ETag and URL-refresh settings.

Remote failures are exposed as `h5rs.S3Error`, an `OSError` subclass.
`kind` identifies errors such as `dns`, `tls`, `connect`, `timeout`,
and `http`. `status_code` carries the unexpected HTTP status, if
available. `reason` describes the cause without including signed URLs.

### Viewer APIs

```python
pool = h5rs.ReadPool(threads=8, concurrent_requests=8, max_s3_gets=32)
file = h5rs.File.open_s3(
    sign_get(), pool, refresh_url=sign_get,
    prefetch_header=True, preload_chunk_index=True, max_gap_bytes=128 * 1024,
)
distance = file["distance"][1000:1500]
channel = file["channel_0"].with_chunk_cache(24 * 1024**2)
tile, overview = h5rs.read_many([
    (channel, (slice(1000, 1500), slice(0, 512))),
    (channel, (slice(1500, 2000, 4), slice(0, 512))),
])
envelope = channel.read_minmax((slice(1000, 1500), slice(0, 512)), block=8)
```

`sign_get` supplies a fresh URL for the same object path. The callback can run
on a worker thread. Keep it independent of h5rs. When a GET receives HTTP 401
or 403, reads sharing the handle reuse a refreshed URL and retry once. The
object size and ETag remain pinned. Credential discovery and URL signing
stay with the caller. Both path-style and virtual-hosted HTTPS URLs work
when the endpoint supports conditional byte ranges.

`max_s3_gets` enables the async HTTP transport for chunks (the
`s3-async-io` Cargo feature). The limit is shared by files using the same
pool. A worker coordinates each batch, and CPU workers decode the results.
HDF5 metadata reads and contiguous spans use the synchronous transport.
Use `open_s3_many` when opening several URLs together. Chunk index preloading
occurs when each dataset is opened and falls back to on-demand lookup if the
RAM budget is reached. Header prefetch and index preloading are independent.

The decoded cache is opt-in and bounded per dataset handle. Clones of a
cached handle share its entries. Reuse the handle to get cache hits. Range
grouping limits `max_range_chunks`, `max_range_bytes` and `max_gap_bytes`
allow extra bytes to be traded for fewer requests. Gaps default to zero.

Steps, reverse slices and min/max operate on the bounding region. They
reduce the returned array size. Min/max appends an axis of length two for
the minimum and maximum of each bin, including the final partial bin.
`dataset.read(selection, dtype="float32")` requests a float32 result.

Since 0.2.1, `dataset.read_mean(selection, block=64, axis=0)` streams source windows
for mean bins. The staging buffer is at most 8 MiB, plus the usual chunk
scratch space and reduced arrays. Means use float64 accumulation and output,
propagate NaNs and include the final partial bin. Bins are anchored at the
start of the requested slice. For a continuous pyramid across files, align
selections to the same logical bin boundaries or combine partial sums using
their sample counts in the caller.

Use the transport and mean benchmark to measure TLS handshakes and
incremental peak memory with a local HTTPS fixture. It requires OpenSSL
to generate a temporary test CA.

```sh
python benches/s3_mean.py target/s3-mean.json
```

Run the S3 viewer benchmark from a development environment. It uses two
synthetic HDF5 files, five channels, a local HTTP Range server and a
configurable delay on each response. The trace measures independent
optimizations and three concurrent viewers sharing handles.

```sh
python benches/viewer_s3.py target/viewer-s3.json --frames 12 --delay-ms 3
```

The GitHub and PyPI release procedure is in `RELEASE.md` in the repository.

## Python benchmark for the 50 GB distance axis

The generator writes 50 approximately 1 GB shards by default. Each holds
four fully allocated uncompressed 2D datasets of shape `(122112, 1024)`
with `uint16` samples and `(256, 256)` chunks. The distance axis is the
first dimension. Each file stores its starting distance and 0.25 m spacing.
The separate `index.h5` lists filenames, half-open global row and distance
intervals and the physical file sizes.

```sh
python -m pip install "h5py>=3.16,<4"
python benches/distance_50gb/generate.py target/distance-50gb
python benches/distance_50gb/benchmark.py target/distance-50gb/index.h5 target/distance-50gb/results.json
```

The script loads `index.h5` once with h5py before timing. Both backends use
the same in-memory distance lookup and concatenate NumPy arrays for
cross-file requests. The benchmark verifies 99 selections
including all 49 file boundaries, times 4,096 random 512 × 512 tiles
per configuration and scans the full logical payload with each library.
File and dataset open times are recorded separately. OS cache state is
uncontrolled.

To compare scan concurrency and simulate a viewer on the same corpus:

```sh
python benches/distance_50gb/viewer.py target/distance-50gb/index.h5 target/distance-50gb/results-viewer.json
```

Each library runs with 1, 4 and 8 Python threads. The h5rs worker pool has
eight Rust workers in every configuration. Every full scan reads all four
channels in each file with 4,096-row requests. The threaded runs submit
requests in scan order with at most two outstanding requests per Python
thread.

A viewport contains eight 512 × 512 tiles, with two tiles per channel.
The scroll simulation runs five left-to-right traces at 60 Hz and prefetches
up to three future views. The visible range moves by 256 rows per frame,
so consecutive viewports overlap. Scroll latency measures how long a
viewport's data is late relative to the scheduled frame. Jump latency
measures the time until all eight tiles are ready after each of 256
random jumps. The simulation measures the data path through NumPy.

To simulate two visible datasets with one full-width request per dataset:

```sh
python benches/distance_50gb/viewer.py target/distance-50gb/index.h5 target/distance-50gb/results-viewer-datasets-2.json --datasets 2 --request-columns 1024 --python-threads 1 2 --viewer-only --warm-jumps
```

With one Python thread the two requests run in sequence. With two threads
both can be in flight together. For jump measurements, `--warm-jumps`
replays the same trace once per configuration before timing.

To measure tiles of approximately 10 MB with 1, 2 and 8 simultaneous
Python callers:

```sh
python benches/distance_50gb/large_tiles.py target/distance-50gb/index.h5 target/distance-50gb/results-10mb.json
```

The default tile contains 4,864 × 1,024 `uint16` values. Each configuration
replays the same 512 selections once before timing two further replays.
Use `--backend h5py` for the reference reader.

## Local tile benchmark

```sh
python benches/make_fixture.py tiles.h5
cargo bench --locked --bench read_tiles -- tiles.h5
cargo bench --locked --features simd-deflate --bench read_tiles -- tiles.h5
```

The benchmark outputs one JSON record per combination of decoder, compression
pipeline, tile size and worker count. It measures warmed local reads by one
client on synthetic 4096 × 4096 numerical datasets with 256 × 256 chunks.
The default Deflate backend is miniz_oxide. The `simd-deflate` feature selects
zlib-rs with runtime CPU feature detection for a direct comparison.

One Windows x64 run on an AMD Ryzen 7 9800X3D used 256 warmed requests per
combination, Rust 1.95.0, bundled HDF5 2.2.0 and gzip level 4. These were the
results for 512 × 512 tiles crossing nine Shuffle + Deflate chunks.

| Backend | Workers | Tiles/s | p95 |
| --- | ---: | ---: | ---: |
| miniz_oxide | 1 | 741 | 1.55 ms |
| miniz_oxide | 8 | 1,928 | 0.76 ms |
| zlib-rs | 1 | 1,037 | 1.32 ms |
| zlib-rs | 8 | 2,351 | 0.65 ms |

The comparison covers the full read path and changes the Deflate backend.
It does not isolate the speedup due to SIMD instructions.
