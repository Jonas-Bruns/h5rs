"""Generate small HDF5 files and read them back with h5py as the oracle."""

import random
import shutil
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np


def make_fixtures(root):
    path = root / "reference.h5"
    with h5py.File(path, "w", userblock_size=128 * 1024) as file:
        grid = np.arange(11 * 17).reshape(11, 17) * 17 - 950
        cube = np.arange(7 * 11 * 9).reshape(7, 11, 9) / 8 - 16
        file.create_dataset("raw", data=grid.astype("<i4"), chunks=(4, 6))
        file.create_dataset("gzip", data=cube.astype("<f8"), chunks=(3, 4, 5), compression="gzip")
        raw3d = file.create_dataset("raw3d", cube.shape, dtype="<f4", chunks=(3, 4, 5))
        for depth in (6, 3, 0):
            raw3d[depth:depth + 3] = cube[depth:depth + 3]
        mixed = file.create_dataset(
            "mixed", shape=(11, 17), dtype=">i2", chunks=(4, 6),
            shuffle=True, compression="gzip", compression_opts=6, fillvalue=-2345,
        )
        mixed[:4, :6] = grid[:4, :6]
        # Skip both filters, then skip only shuffle. Remaining chunks are unallocated.
        mixed.id.write_direct_chunk((4, 6), grid[4:8, 6:12].astype(">i2"), filter_mask=3)
        mixed.id.write_direct_chunk(
            (4, 12), zlib.compress(grid[:4, :6].astype(">i2").tobytes()), filter_mask=1
        )
        shuffled = np.ascontiguousarray(grid[:4, :6].astype(">i2").view("u1").reshape(-1, 2).T)
        mixed.id.write_direct_chunk((8, 0), shuffled, filter_mask=2)

        for tag in ("i1", "u1", "<i2", ">u2", ">i4", "<u4", ">i8", "<u8", ">f4", "<f8"):
            file.create_dataset(
                "type-" + tag.replace("<", "le").replace(">", "be"),
                data=np.arange(63).astype(tag).reshape(7, 9), chunks=(3, 4), shuffle=True,
            )
        file["soft"] = h5py.SoftLink("/raw")
        file["external"] = h5py.ExternalLink("external.h5", "/raw")
        file.create_dataset("fletcher", data=grid, chunks=(4, 6), fletcher32=True)
        file.create_dataset("text", shape=(2, 3), dtype="S4", chunks=(2, 3))
        file.create_dataset("contiguous", data=grid)
        file.create_dataset("rank4", shape=(1, 1, 1, 1), dtype="u1", chunks=(1, 1, 1, 1))
        file.create_dataset("empty", shape=(0, 7), maxshape=(None, 7), dtype="u1", chunks=(2, 3))
        # 2 TiB of logical data; the fixture contains one four-byte chunk.
        high = file.create_dataset("high", shape=(2**40, 2), dtype="u1", chunks=(2, 2))
        high[-2:, :] = [[7, 8], [9, 10]]
        file.create_dataset("never", shape=(4, 4), dtype="u1", chunks=(2, 2), fill_time="never")

    with h5py.File(root / "external.h5", "w") as external:
        external.create_dataset("raw", data=grid + 1, chunks=(4, 6))

    # Distinguish raw-data corruption from metadata or file-open failures.
    corrupt = root / "corrupt.h5"
    with h5py.File(corrupt, "w", libver="latest") as file:
        dataset = file.create_dataset("data", data=grid, chunks=(4, 6), compression="gzip")
        offset = dataset.id.get_chunk_info_by_coord((0, 0)).byte_offset
    with corrupt.open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"\x00\x00")

    truncated = root / "truncated.h5"
    shutil.copyfile(path, truncated)
    with truncated.open("r+b") as stream:
        stream.truncate(1024)

    # Test an object index written with newer HDF5 format bounds.
    with h5py.File(root / "latest.h5", "w", libver="latest") as file:
        dataset = file.create_dataset(
            "extendible", shape=(13, 7), maxshape=(None, 7), dtype=">f4",
            chunks=(3, 4), fillvalue=1.25, compression="gzip",
        )
        dataset[:5, :6] = grid[:5, :6]

    # A dense dataset beyond the address-index limit. All selected chunks
    # must still be readable through coordinate lookup after preloading stops.
    with h5py.File(root / "large-index.h5", "w", libver="latest") as file:
        values = np.arange(257 * 256, dtype="<u4").reshape(257, 256)
        file.create_dataset("large", data=values, chunks=(1, 1))


def write_oracle(root):
    rng = random.Random(771)
    with (root / "cases.tsv").open("w", encoding="utf-8") as cases:
        def vector(values):
            return ",".join(map(str, values))

        def record(file, name, bounds):
            dataset = file[name]
            selection = tuple(slice(start, end) for start, end in bounds)
            expected = dataset[selection]
            case = record.count
            record.count += 1
            (root / f"{case}.bin").write_bytes(expected.tobytes(order="C"))
            cases.write("\t".join([
                Path(file.filename).name, name, expected.dtype.str, vector(dataset.shape),
                vector(dataset.chunks), vector(expected.shape),
                vector(start for start, _ in bounds), vector(end for _, end in bounds), str(case),
            ]) + "\n")

        record.count = 0
        with h5py.File(root / "reference.h5", "r") as file:
            for name in ("raw", "gzip", "mixed", "soft"):
                record(file, name, [(0, extent) for extent in file[name].shape])
                record(file, name, [(1, extent - 1) for extent in file[name].shape])
                for _ in range(16):
                    bounds = [sorted(rng.choices(range(extent + 1), k=2)) for extent in file[name].shape]
                    record(file, name, bounds)
            for name in file:
                if name.startswith("type-"):
                    record(file, name, [(1, 7), (2, 9)])
            record(file, "raw3d", [(1, 7), (2, 10), (1, 8)])
            record(file, "empty", [(0, 0), (0, 7)])
            record(file, "high", [(2**40 - 3, 2**40), (0, 2)])
        with h5py.File(root / "latest.h5", "r") as file:
            record(file, "extendible", [(1, 13), (2, 7)])


if __name__ == "__main__":
    root = Path(sys.argv[1])
    make_fixtures(root)
    write_oracle(root)
