"""Generate a reproducible, structured numerical tile benchmark input."""

import sys

import h5py
import numpy as np


side = 4096
rng = np.random.default_rng(177)
x = np.arange(side, dtype=np.float32)[None, :]
y = x.T
wave = 1500 + 450 * np.sin(x / 27) + 370 * np.cos((x + y) / 91)
image = (wave + rng.integers(0, 100, (side, side), dtype=np.uint16)).astype("<u2")
with h5py.File(sys.argv[1], "w") as file:
    for name, shuffle in (("deflate", False), ("shuffle_deflate", True)):
        file.create_dataset(
            name, data=image, chunks=(256, 256),
            compression="gzip", compression_opts=4, shuffle=shuffle,
        )
