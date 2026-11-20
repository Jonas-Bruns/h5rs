"""Exercise the installed Python wheel with local and S3-style reads."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import h5py
import h5rs
import numpy as np


def check_reads(file, array_2d, array_3d, *, use_index=False):
    first = file["channel_2d"]
    second = (
        file.dataset_with_s3_index("channel_3d") if use_index else file["channel_3d"]
    )
    assert first.shape == array_2d.shape
    assert first.chunks == (8, 8)
    assert second.dtype == array_3d.dtype
    distance = file["distance"]
    assert distance.chunks is None
    np.testing.assert_array_equal(distance[34:3:-4], np.arange(48)[34:3:-4])
    assert distance.read(slice(0, 12), dtype="float32").dtype == np.dtype("float32")
    with ThreadPoolExecutor(max_workers=2) as executor:
        tile_2d = executor.submit(first.__getitem__, (slice(5, 27), slice(3, 26)))
        tile_3d = executor.submit(
            second.__getitem__, (slice(6, 30), slice(1, 7), slice(2, 11))
        )
        np.testing.assert_array_equal(tile_2d.result(), array_2d[5:27, 3:26])
        np.testing.assert_array_equal(tile_3d.result(), array_3d[6:30, 1:7, 2:11])
    cached = first.with_chunk_cache(8 * 8 * 4 * 4)
    batch = h5rs.read_many([(cached, (slice(0, 12), slice(4, 19))), (distance, slice(2, 21, 3)),
                            (cached, (slice(3, 17), slice(4, 19)))])
    np.testing.assert_array_equal(batch[0], array_2d[0:12, 4:19])
    np.testing.assert_array_equal(batch[1], np.arange(48)[2:21:3])
    np.testing.assert_array_equal(batch[2], array_2d[3:17, 4:19])
    pairs = cached.read_minmax((slice(0, 12), slice(4, 19)), block=3)
    bins = array_2d[0:12, 4:19].reshape(4, 3, 15)
    np.testing.assert_array_equal(pairs, np.stack((bins.min(axis=1), bins.max(axis=1)), axis=-1))


def main():
    array_2d = np.arange(48 * 32, dtype=np.int32).reshape(48, 32)
    array_3d = np.arange(48 * 8 * 12, dtype=np.float32).reshape(48, 8, 12)
    pool = h5rs.ReadPool(threads=2, concurrent_requests=2)
    # On Windows a failing assertion can keep HDF5 handles alive in the traceback.
    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        fixture = root / "reference.h5"
        with h5py.File(fixture, "w", userblock_size=128 * 1024) as file:
            file.create_dataset(
                "channel_2d",
                data=array_2d,
                chunks=(8, 8),
                compression="gzip",
                shuffle=True,
            )
            file.create_dataset("channel_3d", data=array_3d, chunks=(8, 4, 6))
            file.create_dataset("distance", data=np.arange(48, dtype=">i8"))
        check_reads(h5rs.File(fixture, pool), array_2d, array_3d)

        server_script = Path(__file__).with_name("range_server.py")
        with subprocess.Popen([sys.executable, str(server_script), str(root)]) as server:
            try:
                port_file = root / "port"
                deadline = time.monotonic() + 10
                port = ""
                while not port:
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Range server failed to start")
                    if port_file.exists():
                        port = port_file.read_text(encoding="utf-8").strip()
                    if not port:
                        time.sleep(0.025)
                url = f"http://127.0.0.1:{port}/reference.h5"
                check_reads(h5rs.File.open_s3(url, pool, max_gap_bytes=1024), array_2d, array_3d, use_index=True)
                async_pool = h5rs.ReadPool(threads=2, concurrent_requests=2, max_s3_gets=8)
                check_reads(h5rs.File.open_s3(url, async_pool), array_2d, array_3d, use_index=True)
            finally:
                server.terminate()
    print("Installed h5rs passed local and S3-style reads")


if __name__ == "__main__":
    main()
