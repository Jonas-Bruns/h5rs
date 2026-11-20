"""Generate physical HDF5 shards and a separate HDF5 distance index."""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


CHANNELS = tuple(f"channel_{index}" for index in range(4))
COLUMNS = 1024
CHUNKS = (256, 256)
STEP_M = 0.25
ROWS_PER_SHARD = 122_112
FORMAT_VERSION = 1


def generate(directory, shards, rows):
    directory.mkdir(parents=True, exist_ok=True)
    files = [directory / f"distance_{index:03d}.h5" for index in range(shards)]
    expected_payload = rows * COLUMNS * len(CHANNELS) * np.dtype("<u2").itemsize
    pending = []
    for index, path in enumerate(files):
        if path.is_file():
            with h5py.File(path, "r") as existing:
                if (existing.attrs.get("format_version") != FORMAT_VERSION
                        or existing.attrs.get("row_start") != index * rows
                        or existing.attrs.get("row_stop") != (index + 1) * rows
                        or path.stat().st_size < expected_payload):
                    raise ValueError(f"Unexpected existing shard {path}")
        else:
            pending.append((index, path))

    remaining_bytes = (expected_payload + 1024**2) * len(pending)
    if remaining_bytes > shutil.disk_usage(directory).free:
        raise OSError(f"Need approximately {remaining_bytes:,} free bytes for remaining shards")

    for index, path in pending:
        start = index * rows
        temporary = path.with_suffix(".h5.partial")
        with h5py.File(temporary, "w", libver="v110") as file:
            file.attrs.update(
                format_version=FORMAT_VERSION, row_start=start, row_stop=start + rows,
                distance_start_m=start * STEP_M, distance_step_m=STEP_M,
            )
            for channel, name in enumerate(CHANNELS):
                dataset = file.create_dataset(name, (rows, COLUMNS), dtype="<u2", chunks=CHUNKS)
                dataset.dims[0].label = "distance"
                dataset.dims[1].label = "sample"
                rng = np.random.default_rng([1739, index, channel])
                for first in range(0, rows, 16 * CHUNKS[0]):
                    stop = min(first + 16 * CHUNKS[0], rows)
                    dataset[first:stop, :] = rng.integers(
                        0, 2**16, (stop - first, COLUMNS), dtype=np.uint16
                    )
        temporary.replace(path)
        print(f"{index + 1:02}/{shards:02} {path.name} {path.stat().st_size:,} bytes", flush=True)

    starts = np.arange(shards, dtype="<u8") * rows
    stops = starts + rows
    temporary = directory / "index.h5.partial"
    with h5py.File(temporary, "w") as index:
        index.attrs.update(format_version=FORMAT_VERSION, distance_step_m=STEP_M)
        index["file"] = np.array([path.name for path in files], dtype=h5py.string_dtype("utf-8"))
        index["row_start"] = starts
        index["row_stop"] = stops
        index["distance_start_m"] = starts * STEP_M
        index["distance_stop_m"] = stops * STEP_M
        index["file_size_bytes"] = np.array([path.stat().st_size for path in files], dtype="<u8")
    destination = directory / "index.h5"
    temporary.replace(destination)
    print(f"{destination} | {sum(path.stat().st_size for path in files):,} shard bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--shards", type=int, default=50)
    parser.add_argument("--rows", type=int, default=ROWS_PER_SHARD)
    options = parser.parse_args()
    if options.shards < 2 or options.rows < 1024 or options.rows % CHUNKS[0]:
        parser.error("use at least two shards and a row count divisible by 256 and at least 1024")
    generate(options.directory, options.shards, options.rows)
