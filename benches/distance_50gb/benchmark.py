"""Read a sharded HDF5 distance axis with h5py and h5rs and record timings."""

import argparse
import json
import platform
import random
import sys
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, process_time

import h5py
import h5rs
import numpy as np

from generate import CHANNELS, CHUNKS, COLUMNS


class DistanceIndex:
    def __init__(self, path):
        with h5py.File(path, "r") as index:
            self.step = float(index.attrs["distance_step_m"])
            self.files = [path.parent / name for name in index["file"].asstr()[:]]
            self.starts = index["row_start"][:].tolist()
            self.stops = index["row_stop"][:].tolist()
            self.sizes = index["file_size_bytes"][:].tolist()
            np.testing.assert_array_equal(index["distance_start_m"][:], np.asarray(self.starts) * self.step)
            np.testing.assert_array_equal(index["distance_stop_m"][:], np.asarray(self.stops) * self.step)
        if (not self.files or self.starts != [0, *self.stops[:-1]]
                or len(self.files) != len(self.stops) or len(self.files) != len(self.sizes)
                or any(start >= stop for start, stop in zip(self.starts, self.stops))):
            raise ValueError("index does not describe contiguous shards")
        if any(path.stat().st_size != size for path, size in zip(self.files, self.sizes)):
            raise ValueError("shard size differs from the HDF5 index")

    def split(self, start_m, stop_m):
        first = start_m / self.step
        last = stop_m / self.step
        if not (first.is_integer() and last.is_integer() and 0 <= first < last <= self.stops[-1]):
            raise ValueError("use a nonempty in-bounds distance range aligned to the sample grid")
        row, stop = int(first), int(last)
        shard = bisect_right(self.stops, row)
        while row < stop:
            end = min(stop, self.stops[shard])
            yield shard, slice(row - self.starts[shard], end - self.starts[shard])
            row = end
            shard += 1


class Reader:
    def __init__(self, index, backend, native_workers, callers):
        self.index = index
        start = perf_counter()
        if backend == "h5py":
            self.files = [h5py.File(path, "r") for path in index.files]
        else:
            pool = h5rs.ReadPool(native_workers, callers)
            self.files = [h5rs.File(path, pool) for path in index.files]
        self.datasets = [tuple(file[name] for name in CHANNELS) for file in self.files]
        self.open_seconds = perf_counter() - start

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.datasets.clear()
        for file in self.files:
            if isinstance(file, h5py.File):
                file.close()
        self.files.clear()

    def read(self, request):
        channel, start_m, stop_m, samples = request
        parts = [
            self.datasets[shard][channel][rows, samples]
            for shard, rows in self.index.split(start_m, stop_m)
        ]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


def tile(index, start_row, channel, sample, size=512):
    return channel, start_row * index.step, (start_row + size) * index.step, slice(sample, sample + size)


def random_tiles(index, count):
    rng = random.Random(5615)
    for request in range(count):
        if request % 4 == 0:
            boundary = index.stops[request // 4 % (len(index.files) - 1)]
            row = boundary - rng.randrange(1, 512)
        else:
            shard = rng.randrange(len(index.files))
            row = rng.randint(index.starts[shard], index.stops[shard] - 512)
        yield tile(index, row, rng.randrange(len(CHANNELS)), rng.randrange(COLUMNS - 511))


def verify(index):
    with Reader(index, "h5py", 1, 1) as oracle, Reader(index, "h5rs", 4, 1) as reader:
        expected = oracle.datasets[0][0]
        actual = reader.datasets[0][0]
        assert (actual.shape, actual.chunks, actual.dtype) == (expected.shape, expected.chunks, expected.dtype)
        np.testing.assert_array_equal(actual[-12:, :13], expected[-12:, :13], strict=True)
        requests = [tile(index, start + 13, shard % len(CHANNELS), 19)
                    for shard, start in enumerate(index.starts)]
        requests.extend(tile(index, boundary - 317, shard % len(CHANNELS), 31)
                        for shard, boundary in enumerate(index.stops[:-1]))
        for request in requests:
            np.testing.assert_array_equal(reader.read(request), oracle.read(request), strict=True)
    return len(requests)


def sample(data):
    return int(data[0, 0]) + int(data[data.shape[0] // 2, data.shape[1] // 2]) + int(data[-1, -1])


def timed_tiles(index, requests, backend, callers, workers):
    def run(reader, selections):
        times = []
        size = checksum = 0
        for request in selections:
            start = perf_counter()
            result = reader.read(request)
            times.append(perf_counter() - start)
            size += result.nbytes
            checksum += sample(result)
        return times, size, checksum

    with Reader(index, backend, workers, callers) as reader:
        run(reader, requests[:64])
        start, cpu = perf_counter(), process_time()
        if callers == 1:
            results = [run(reader, requests)]
        else:
            with ThreadPoolExecutor(max_workers=callers) as pool:
                results = list(pool.map(lambda group: run(reader, group), (requests[i::callers] for i in range(callers))))
        elapsed, cpu = perf_counter() - start, process_time() - cpu
        timings = [latency for batch, _, _ in results for latency in batch]
        size, checksum = sum(row[1] for row in results), sum(row[2] for row in results)
        return metrics("random_512", backend, callers, workers, elapsed, cpu,
                       reader.open_seconds, size, checksum, timings)


def scan(index, backend, workers):
    with Reader(index, backend, workers, 1) as reader:
        size = checksum = 0
        timings = []
        start, cpu = perf_counter(), process_time()
        for shard, (start_row, stop_row) in enumerate(zip(index.starts, index.stops)):
            for channel in range(len(CHANNELS)):
                for first in range(start_row, stop_row, 4096):
                    request = (channel, first * index.step, min(first + 4096, stop_row) * index.step, slice(None))
                    request_start = perf_counter()
                    result = reader.read(request)
                    timings.append(perf_counter() - request_start)
                    size += result.nbytes
                    checksum += sample(result)
            if (shard + 1) % 10 == 0:
                print(f"{backend} full scan {shard + 1}/{len(index.files)} shards", file=sys.stderr, flush=True)
        elapsed, cpu = perf_counter() - start, process_time() - cpu
        return metrics("full_scan", backend, 1, workers, elapsed, cpu,
                       reader.open_seconds, size, checksum, timings)


def metrics(workload, backend, callers, workers, elapsed, cpu, open_seconds, size, checksum, times):
    p50, p95, p99 = (np.percentile(times, [50, 95, 99]) * 1000).tolist()
    return dict(workload=workload, backend=backend, python_callers=callers,
                h5rs_workers=workers if backend == "h5rs" else None, requests=len(times),
                seconds=elapsed, cpu_seconds=cpu, open_seconds=open_seconds, output_bytes=size,
                logical_gb_per_second=size / elapsed / 1e9, requests_per_second=len(times) / elapsed,
                p50_ms=p50, p95_ms=p95, p99_ms=p99, sampled_checksum=checksum)


def benchmark(index_path, output, count, threads):
    index = DistanceIndex(index_path)
    verified = verify(index)
    print(f"{verified} selections verified against h5py", flush=True)
    configurations = [("h5py", 1, 1), ("h5rs", 1, 1), ("h5rs", 1, threads),
                      ("h5rs", threads, threads), ("h5py", threads, 1)]
    report = {
        "environment": {"platform": platform.platform(), "cpu": platform.processor(),
                        "python": platform.python_version(), "h5py": h5py.__version__,
                        "h5py_hdf5": h5py.version.hdf5_version, "numpy": np.__version__,
                        "h5rs": version("h5rs")},
        "dataset": {"files": len(index.files), "shard_disk_bytes": sum(index.sizes),
                    "channels_per_shard": len(CHANNELS), "chunks": CHUNKS, "dtype": "<u2",
                    "compression": None, "distance_step_m": index.step,
                    "distance_interval_m": [0, index.stops[-1] * index.step]},
        "method": {"verified_tiles": verified, "cross_file_tiles": (count + 3) // 4,
                   "os_cache": "uncontrolled", "tile_warmup_requests": 64,
                   "latency": "request service time, excluding executor queue"},
        "results": [],
    }
    requests = list(random_tiles(index, count))

    def save(result):
        previous = next((entry for entry in report["results"] if entry["workload"] == result["workload"]), None)
        if previous:
            assert result["sampled_checksum"] == previous["sampled_checksum"], result
            assert result["output_bytes"] == previous["output_bytes"], result
        report["results"].append(result)
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(json.dumps(result), flush=True)

    for backend, callers, workers in configurations:
        save(timed_tiles(index, requests, backend, callers, workers))
    for backend, workers in (("h5py", 1), ("h5rs", threads)):
        save(scan(index, backend, workers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--requests", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    options = parser.parse_args()
    if options.requests < 64 or options.threads < 2:
        parser.error("use at least 64 requests and 2 threads")
    benchmark(options.index, options.output, options.requests, options.threads)
