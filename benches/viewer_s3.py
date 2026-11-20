"""Replay five-channel viewports over a local S3-style HTTP range endpoint."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
from statistics import median, quantiles
from tempfile import TemporaryDirectory
from threading import Lock, Thread
from time import perf_counter, sleep
from urllib.parse import urlsplit

import h5py
import h5rs
import numpy as np


ROWS = 16384
TILE_ROWS = 1024
CHANNELS = (
    (0, "a", (3072,), (512,), (slice(0, 2560),)),
    (0, "b", (3072,), (512,), (slice(0, 2560),)),
    (0, "c", (64, 48), (32, 8), (slice(None), slice(0, 40))),
    (0, "d", (64, 48), (32, 8), (slice(None), slice(0, 40))),
    (1, "e", (64, 48), (32, 8), (slice(None), slice(0, 40))),
)


def make_files(root):
    rng = np.random.default_rng(9)
    paths = [root / f"{index}.h5" for index in range(2)]
    for index, path in enumerate(paths):
        with h5py.File(path, "w") as file:
            for part, name, tail, chunks, _ in CHANNELS:
                if index == part:
                    values = rng.integers(-1000, 1000, (ROWS, *tail), dtype=np.int16)
                    file.create_dataset(name, data=values.astype("f4"), chunks=(64, *chunks),
                                        compression="gzip", compression_opts=1, shuffle=True)
            if index == 0:
                values = np.arange(752_000, dtype=">f8")
                file.create_dataset("distance", data=values)
                file.create_dataset("distance_chunked", data=values, chunks=(16384,))
                file.create_dataset("contiguous", data=np.arange(3_072 * 2560, dtype="f4").reshape(3_072, 2560))
                compact = values[:256]
                properties = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
                properties.set_layout(h5py.h5d.COMPACT)
                dataset = h5py.h5d.create(file.id, b"compact", h5py.h5t.py_create(compact.dtype),
                                          h5py.h5s.create_simple(compact.shape), dcpl=properties)
                dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, compact)
    return paths


class Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True

    def __init__(self, paths, delay):
        self.lock = Lock()
        self.paths = {f"/{i}": path for i, path in enumerate(paths)}
        self.delay = delay
        self.expired = False
        self.active = 0
        self.reset()
        super().__init__(("127.0.0.1", 0), RangeHandler)

    def reset(self):
        with self.lock:
            self.stats = dict(gets=0, bytes=0, peak=0, rejected=0)


class RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = self.server.paths[parsed.path]
        size = path.stat().st_size
        etag = '"changed"' if parsed.query == "token=changed" else f'"viewer-{path.stem}"'
        start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
        length = min(end + 1, size) - start
        with self.server.lock:
            stats = self.server.stats
            stats["gets"] += 1
            expired = self.server.expired and parsed.query == "token=old"
            if expired:
                stats["rejected"] += 1
            else:
                self.server.active += 1
                stats["peak"] = max(stats["peak"], self.server.active)
        if expired:
            self.send_error(403)
            return
        try:
            if self.headers.get("If-Match", etag) != etag:
                self.send_error(412)
                return
            sleep(self.server.delay)
            with path.open("rb") as source:
                source.seek(start)
                body = source.read(length)
            with self.server.lock:
                stats["bytes"] += len(body)
            self.send_response(206)
            self.send_header("ETag", etag)
            self.send_header("Content-Range", f"bytes {start}-{start + length - 1}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            pass
        finally:
            with self.server.lock:
                self.server.active -= 1


@contextmanager
def endpoint(paths, delay):
    with Server(paths, delay) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host = f"http://127.0.0.1:{server.server_port}"
            yield server, [f"{host}/{i}" for i in range(len(paths))]
        finally:
            server.shutdown()
            thread.join()


def measure(calls, server=None):
    samples = []
    if server is not None:
        server.reset()
    for call in calls:
        start = perf_counter()
        call()
        samples.append((perf_counter() - start) * 1000)
    stats = {
        "samples": len(samples), "median_ms": median(samples),
        "p95_ms": quantiles(samples, n=100, method="inclusive")[94] if len(samples) > 1 else samples[0],
        "total_s": sum(samples) / 1000,
    }
    return stats | (server.stats.copy() if server is not None else {})


def compare_layouts(path):
    layout_reads = {
        "distance_1d_contiguous": ("distance", slice(1009, 263153)),
        "distance_1d_chunked": ("distance_chunked", slice(1009, 263153)),
        "contiguous_2d": ("contiguous", (slice(333, 1357), slice(None))),
        "compact_1d": ("compact", slice(32, 224)),
    }
    results = {}
    with h5py.File(path) as reference:
        reader = h5rs.File(path)
        for name, (dataset, selection) in layout_reads.items():
            implementations = {"h5py": reference[dataset], "h5rs": reader[dataset]}
            np.testing.assert_array_equal(implementations["h5py"][selection], implementations["h5rs"][selection])
            results[name] = {
                api: measure([lambda: handle[selection]] * 48)
                for api, handle in implementations.items()
            }
    return results


def benchmark_viewer(server, urls, frames, published=False):
    rng = random.Random(42)
    workloads = {
        "scroll": [231 + frame * 48 for frame in range(frames)],
        "jumps": [rng.randrange(ROWS - TILE_ROWS) for _ in range(frames)],
    }
    settings = [
        ("baseline", None, 0, False, False),
        ("decoded_cache", None, 0, True, False),
        ("gaps_128KiB", None, 128 * 1024, False, False),
        *[(f"async_{limit}", limit, 0, False, False) for limit in (8, 16, 32, 64)],
        ("batch", None, 0, False, True),
        ("cache_gap_async32", 32, 128 * 1024, True, False),
    ]
    if published:
        settings = settings[:1]
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for name, limit, gap, cache, batch in settings:
            traces = {}
            for workload, starts in workloads.items():
                if published:
                    pool = h5rs.ReadPool(8, 8)
                    files = [h5rs.File.open_s3(url, pool) for url in urls]
                else:
                    pool = h5rs.ReadPool(8, 8, max_s3_gets=limit)
                    files = [h5rs.File.open_s3(url, pool, max_gap_bytes=gap) for url in urls]
                handles = [(files[part].dataset_with_s3_index(channel), tail) for part, channel, _, _, tail in CHANNELS]
                if cache:
                    handles = [(ds.with_chunk_cache(24 * 2**20), tail) for ds, tail in handles]

                def viewport(row):
                    requests = [(ds, (slice(row, row + TILE_ROWS), *tail)) for ds, tail in handles]
                    if batch:
                        arrays = h5rs.read_many(requests)
                    else:
                        pending = [executor.submit(ds.__getitem__, selection) for ds, selection in requests]
                        arrays = [future.result() for future in pending]
                    assert sum(array.nbytes for array in arrays) == 50 * 2**20

                traces[workload] = measure([lambda row=row: viewport(row) for row in starts], server)
            results[name] = traces
    return results


def benchmark_concurrent_viewers(server, urls, frames):
    variants = {
        "baseline": (None, False),
        "decoded_cache": (None, True),
        "async_32": (32, False),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=15) as executor:
        for name, (limit, cache) in variants.items():
            pool = h5rs.ReadPool(8, 8, max_s3_gets=limit)
            files = [h5rs.File.open_s3(url, pool) for url in urls]
            handles = [(files[part].dataset_with_s3_index(channel), tail) for part, channel, _, _, tail in CHANNELS]
            if cache:
                handles = [(ds.with_chunk_cache(64 * 2**20), tail) for ds, tail in handles]

            def read_frame_set(frame):
                work = [
                    executor.submit(ds.__getitem__, (slice(row, row + TILE_ROWS), *tail))
                    for row in (409 + 48 * frame, 5675 + 48 * frame, 11201 + 48 * frame)
                    for ds, tail in handles
                ]
                assert sum(item.result().nbytes for item in work) == 150 * 2**20

            results[name] = measure([lambda frame=frame: read_frame_set(frame) for frame in range(frames)], server)
    return results


def benchmark_open(server, urls):
    def open_then_read(prefetch=False, parallel=False, preload=False):
        if parallel:
            files = h5rs.open_s3_many(urls, prefetch_header=prefetch, preload_chunk_index=preload)
        else:
            files = [h5rs.File.open_s3(url, prefetch_header=prefetch, preload_chunk_index=preload) for url in urls]
        for part, name, _, _, tail in CHANNELS:
            files[part][name][slice(768, 768 + TILE_ROWS), *tail]

    def open_only(prefetch=False, parallel=False):
        if parallel:
            return h5rs.open_s3_many(urls, prefetch_header=prefetch)
        return [h5rs.File.open_s3(url, prefetch_header=prefetch) for url in urls]

    return {
        "serial_open": measure([open_only] * 8, server),
        "prefetch_open": measure([lambda: open_only(prefetch=True)] * 8, server),
        "prefetch_and_parallel_open": measure([lambda: open_only(prefetch=True, parallel=True)] * 8, server),
        "first_viewport": measure([open_then_read] * 3, server),
        "first_viewport_preload_index": measure([lambda: open_then_read(preload=True)] * 3, server),
    }


def benchmark_refresh(server, urls):
    old = urls[0] + "?token=old"
    new = urls[0] + "?token=new"
    selection = (slice(500, 1524), slice(0, 2560))
    results = {}
    for name, limit in (("blocking", None), ("async", 16)):
        calls = []
        pool = h5rs.ReadPool(8, 8, max_s3_gets=limit)

        def refresh():
            calls.append(1)
            return new

        server.expired = False
        dataset = h5rs.File.open_s3(old, pool, refresh_url=refresh, preload_chunk_index=True)["a"]
        reference = dataset[selection]
        server.expired = True
        try:
            def read_refreshed():
                np.testing.assert_array_equal(dataset[selection], reference)

            results[name] = measure([read_refreshed], server) | {"refresh_calls": len(calls)}
            assert len(calls) == 1 and results[name]["rejected"] >= 1
        finally:
            server.expired = False
    return results


def benchmark_transforms(path):
    file = h5rs.File(path)
    dataset = file["a"]
    distance = file["distance"]
    window = (slice(512, 1536), slice(0, 2560))
    block = 16

    def numpy_minmax():
        values = dataset[window].reshape(-1, block, 2560)
        return np.stack((values.min(axis=1), values.max(axis=1)), axis=-1)

    cases = {
        "step_reverse": (
            lambda: dataset[window][::-7, ::5].copy(),
            lambda: dataset[1535:511:-7, 0:2560:5],
        ),
        "minmax": (numpy_minmax, lambda: dataset.read_minmax(window, block)),
        "float32": (
            lambda: distance[1009:263153].astype("f4"),
            lambda: distance.read(slice(1009, 263153), dtype="float32"),
        ),
    }
    results = {}
    for name, (numpy, rust) in cases.items():
        np.testing.assert_array_equal(numpy(), rust())
        results[name] = {
            "read_plus_numpy": measure([numpy] * 16),
            "rust": measure([rust] * 16),
            "output_bytes": rust().nbytes,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=8, choices=range(2, 25))
    parser.add_argument("--delay-ms", type=float, default=3)
    parser.add_argument("--published-baseline", action="store_true", help="Run only the viewer trace using h5rs 0.1.0")
    args = parser.parse_args()
    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        paths = make_files(Path(directory))
        with endpoint(paths, args.delay_ms / 1000) as (server, urls):
            results = {
                "settings": {"delay_ms": args.delay_ms, "frames": args.frames, "rows": ROWS, "viewport_mib": 50},
            }
            if not args.published_baseline:
                results.update(
                    layouts=compare_layouts(paths[0]),
                    refresh=benchmark_refresh(server, urls),
                    open=benchmark_open(server, urls),
                    transforms=benchmark_transforms(paths[0]),
                )
            results["viewports"] = benchmark_viewer(server, urls, args.frames, args.published_baseline)
            if not args.published_baseline:
                results["three_viewers"] = benchmark_concurrent_viewers(server, urls, args.frames)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
