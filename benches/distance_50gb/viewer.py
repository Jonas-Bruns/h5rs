"""Benchmark threaded scans and simulate scrolling and jumps along the distance axis."""

import argparse
import json
import platform
import random
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path
from time import perf_counter, process_time, sleep

from benchmark import DistanceIndex, Reader, metrics, sample, verify
from generate import CHANNELS, COLUMNS


TILE_ROWS = 512
TILE_COLUMNS = 512
SCROLL_STEP = 256
PREFETCH_FRAMES = 3


def observe(reader, request):
    start = perf_counter()
    array = reader.read(request)
    end = perf_counter()
    return array.nbytes, sample(array), end, end - start


def bounded_map(executor, read, requests, window):
    iterator = iter(requests)
    pending = deque(executor.submit(read, request) for request in islice(iterator, window))
    for request in iterator:
        yield pending.popleft().result()
        pending.append(executor.submit(read, request))
    while pending:
        yield pending.popleft().result()


def scan_blocks(index):
    for start, stop in zip(index.starts, index.stops):
        for channel in range(len(CHANNELS)):
            for row in range(start, stop, 4096):
                yield channel, row * index.step, min(row + 4096, stop) * index.step, slice(None)


def scan(index, backend, callers, workers):
    with Reader(index, backend, workers, callers) as reader:
        def read(request):
            return observe(reader, request)

        def measure(results):
            timings = []
            size = checksum = 0
            for number, (used, digest, _, duration) in enumerate(results, 1):
                timings.append(duration)
                size += used
                checksum += digest
                if number % 1200 == 0:
                    print(f"{backend} scan {callers} Python threads: {number} blocks", file=sys.stderr, flush=True)
            return size, checksum, timings

        start, cpu = perf_counter(), process_time()
        if callers == 1:
            size, checksum, timings = measure(map(read, scan_blocks(index)))
        else:
            with ThreadPoolExecutor(max_workers=callers) as executor:
                size, checksum, timings = measure(bounded_map(executor, read, scan_blocks(index), 2 * callers))
        return metrics("full_scan", backend, callers, workers, perf_counter() - start,
                       process_time() - cpu, reader.open_seconds, size, checksum, timings)


def viewport(index, row, *, datasets, request_columns):
    start, stop = row * index.step, (row + TILE_ROWS) * index.step
    return tuple(
        (channel, start, stop, slice(column, column + request_columns))
        for channel in range(datasets)
        for column in range(0, COLUMNS, request_columns)
    )


def scroll_traces(index, frames):
    # Five traces, in increasing distance order, near distributed file boundaries.
    boundaries = index.stops[:-1]
    offsets = sorted({round(i * (len(boundaries) - 1) / 4) for i in range(5)})
    max_start = index.stops[-1] - TILE_ROWS - (frames - 1) * SCROLL_STEP
    if max_start < 0:
        raise ValueError("scroll trace is longer than the dataset")
    traces = []
    for offset in offsets:
        first = max(0, min(boundaries[offset] - (frames // 2) * SCROLL_STEP + 73, max_start))
        traces.append([first + i * SCROLL_STEP for i in range(frames)])
    return traces


def random_jumps(index, count):
    rng = random.Random(9104)
    boundaries = index.stops[:-1]
    return [
        rng.choice(boundaries) - rng.randrange(1, TILE_ROWS)
        if jump % 4 == 0 else rng.randint(0, index.stops[-1] - TILE_ROWS)
        for jump in range(count)
    ]


def launch_viewport(executor, reader, requests):
    return [executor.submit(observe, reader, request) for request in requests]


def finish_viewport(futures):
    tiles = [future.result() for future in futures]
    return sum(tile[0] for tile in tiles), sum(tile[1] for tile in tiles), max(tile[2] for tile in tiles)


def scroll(index, backend, callers, workers, traces, fps, requests_at):
    waits = []
    size = checksum = 0
    frame_period = 1 / fps
    with Reader(index, backend, workers, callers) as reader, ThreadPoolExecutor(max_workers=callers) as executor:
        cpu, start = process_time(), perf_counter()
        for trace in traces:
            origin = perf_counter()
            rows = iter(trace)
            pending = deque(launch_viewport(executor, reader, requests_at(row))
                            for row in islice(rows, PREFETCH_FRAMES + 1))
            for number in range(len(trace)):
                due = origin + number * frame_period
                remaining = due - perf_counter()
                if remaining > 0:
                    sleep(remaining)
                used, digest, ready = finish_viewport(pending.popleft())
                waits.append(max(0.0, ready - due))
                size += used
                checksum += digest
                next_row = next(rows, None)
                if next_row is not None:
                    pending.append(launch_viewport(executor, reader, requests_at(next_row)))
        result = metrics("viewer_scroll", backend, callers, workers, perf_counter() - start,
                         process_time() - cpu, reader.open_seconds, size, checksum, waits)
    result["frames_waiting"] = sum(wait > 0 for wait in waits)
    result["frames_over_one_period"] = sum(wait > frame_period for wait in waits)
    result["max_wait_ms"] = max(waits) * 1000
    return result


def jump(index, backend, callers, workers, rows, fps, requests_at, warm):
    latencies = []
    size = checksum = 0
    with Reader(index, backend, workers, callers) as reader, ThreadPoolExecutor(max_workers=callers) as executor:
        if warm:
            for row in rows:
                finish_viewport(launch_viewport(executor, reader, requests_at(row)))
        cpu, start = process_time(), perf_counter()
        for row in rows:
            action = perf_counter()
            used, digest, ready = finish_viewport(launch_viewport(executor, reader, requests_at(row)))
            latencies.append(ready - action)
            size += used
            checksum += digest
        result = metrics("viewer_jump", backend, callers, workers, perf_counter() - start,
                         process_time() - cpu, reader.open_seconds, size, checksum, latencies)
    result["frames_over_one_period"] = sum(latency > 1 / fps for latency in latencies)
    result["max_wait_ms"] = max(latencies) * 1000
    return result


def benchmark(options):
    index = DistanceIndex(options.index)
    traces = scroll_traces(index, options.scroll_frames)
    jumps = random_jumps(index, options.jumps)
    requests_at = partial(viewport, index, datasets=options.datasets, request_columns=options.request_columns)
    verified = verify(index)
    report = {
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "dataset": {"files": len(index.files), "shard_disk_bytes": sum(index.sizes),
                    "distance_step_m": index.step, "channels": CHANNELS},
        "method": {
            "verified_tiles": verified,
            "os_cache": "uncontrolled",
            "scan_block_rows": 4096,
            "scan_submission_order": "file, channel, increasing distance",
            "scan_latency": "read service time",
            "tile_shape": [TILE_ROWS, options.request_columns],
            "tiles_per_viewport": len(requests_at(0)),
            "visible_datasets": CHANNELS[:options.datasets],
            "scroll_target_fps": options.fps,
            "scroll_prefetch_frames": PREFETCH_FRAMES,
            "scroll_step_m": SCROLL_STEP * index.step,
            "scroll_trace_start_m": [trace[0] * index.step for trace in traces],
            "scroll_frames_per_trace": options.scroll_frames,
            "scroll_latency": "wait beyond the requested display time for all viewport tiles",
            "jump_count": len(jumps),
            "jump_warmup_replays": int(options.warm_jumps),
            "jump_cross_file_count": sum(
                len(tuple(index.split(row * index.step, (row + TILE_ROWS) * index.step))) > 1
                for row in jumps
            ),
            "jump_latency": "time from jump to all viewport tiles ready, including scheduling",
        },
        "results": [],
    }

    def save(result):
        previous = next((row for row in report["results"] if row["workload"] == result["workload"]), None)
        if previous:
            for key in ("requests", "sampled_checksum", "output_bytes"):
                if previous[key] != result[key]:
                    raise ValueError(f"{result['workload']} mismatch for {key}")
        report["results"].append(result)
        temporary = options.output.with_suffix(options.output.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(options.output)
        print(json.dumps(result), flush=True)

    configurations = [(backend, callers, 8 if backend == "h5rs" else 1)
                      for callers in options.python_threads for backend in ("h5py", "h5rs")]
    for backend, callers, workers in configurations:
        save(jump(index, backend, callers, workers, jumps, options.fps, requests_at, options.warm_jumps))
        save(scroll(index, backend, callers, workers, traces, options.fps, requests_at))
    if not options.viewer_only:
        for backend, callers, workers in configurations:
            save(scan(index, backend, callers, workers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scroll-frames", type=int, default=96)
    parser.add_argument("--jumps", type=int, default=256)
    parser.add_argument("--datasets", type=int, choices=range(1, len(CHANNELS) + 1), default=len(CHANNELS))
    parser.add_argument("--request-columns", type=int, default=TILE_COLUMNS)
    parser.add_argument("--python-threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--viewer-only", action="store_true")
    parser.add_argument("--warm-jumps", action="store_true", help="replay the jump trace once before timing")
    options = parser.parse_args()
    if options.fps < 1 or options.scroll_frames < 2 or options.jumps < 1:
        parser.error("use positive fps and jump counts and at least two scroll frames")
    if options.request_columns < 1 or COLUMNS % options.request_columns:
        parser.error(f"--request-columns must divide the {COLUMNS}-column viewport")
    if min(options.python_threads) < 1 or len(set(options.python_threads)) != len(options.python_threads):
        parser.error("--python-threads must contain distinct positive counts")
    benchmark(options)
