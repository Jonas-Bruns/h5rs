"""Measure full-width distance tiles with a configurable row count."""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, process_time

import h5rs
import numpy as np

from benchmark import DistanceIndex, Reader, metrics, sample
from generate import CHANNELS, COLUMNS


def selections(index, rows, count):
    rng = random.Random(8148)
    for number in range(count):
        if number % 4 == 0:
            boundary = index.stops[(number // 4) % (len(index.files) - 1)]
            start = boundary - rng.randrange(1, rows)
        else:
            shard = rng.randrange(len(index.files))
            start = rng.randint(index.starts[shard], index.stops[shard] - rows)
        yield rng.randrange(len(CHANNELS)), start * index.step, (start + rows) * index.step, slice(None)


def verify(index, requests):
    with Reader(index, "h5py", 1, 1) as oracle, Reader(index, "h5rs", 8, 1) as reader:
        for request in requests[:8]:
            np.testing.assert_array_equal(reader.read(request), oracle.read(request), strict=True)


def bench(index, backend, requests, callers):
    with Reader(index, backend, 8, callers) as reader, ThreadPoolExecutor(max_workers=callers) as pool:
        groups = [requests[worker::callers] for worker in range(callers)]

        def read(group):
            times = []
            total = checksum = 0
            for request in group:
                start = perf_counter()
                result = reader.read(request)
                times.append(perf_counter() - start)
                total += result.nbytes
                checksum += sample(result)
            return times, total, checksum

        list(pool.map(read, groups))
        start, cpu = perf_counter(), process_time()
        trials = [list(pool.map(read, groups)) for _ in range(2)]
        elapsed, cpu = perf_counter() - start, process_time() - cpu
        results = [result for trial in trials for result in trial]
        return metrics(
            "large_tiles", backend, callers, 8, elapsed, cpu, reader.open_seconds,
            sum(result[1] for result in results), sum(result[2] for result in results),
            [duration for result in results for duration in result[0]],
        )


def main(options):
    index = DistanceIndex(options.index)
    if not 2 <= options.rows <= min(stop - start for start, stop in zip(index.starts, index.stops)):
        raise ValueError("tile rows must fit in one shard")
    requests = list(selections(index, options.rows, options.requests))
    verify(index, requests)
    report = {
        "label": options.label,
        "h5rs_module": h5rs.__file__,
        "tile_shape": [options.rows, COLUMNS],
        "tile_bytes": options.rows * COLUMNS * 2,
        "dataset_file_bytes": sum(index.sizes),
        "cross_file_tiles_per_trace": (options.requests + 3) // 4,
        "warmup_traces": 1,
        "measured_traces": 2,
        "verified_tiles": min(8, len(requests)),
        "results": [],
    }
    expected_bytes = 2 * len(requests) * report["tile_bytes"]
    for callers in (1, 2, 8):
        result = bench(index, options.backend, requests, callers)
        assert result["output_bytes"] == expected_bytes, result
        if report["results"]:
            assert report["results"][0]["sampled_checksum"] == result["sampled_checksum"], result
        report["results"].append(result)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(options.label, json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", default=4864, type=int)
    parser.add_argument("--requests", default=512, type=int)
    parser.add_argument("--backend", choices=("h5rs", "h5py"), default="h5rs")
    parser.add_argument("--label", default="current")
    options = parser.parse_args()
    if options.requests < 8:
        parser.error("use at least eight requests")
    main(options)
