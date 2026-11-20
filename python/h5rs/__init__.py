"""Parallel read-only access to numerical HDF5 datasets."""

from ._native import Dataset, File, ReadPool, S3Client, S3Error, open_s3_many, read_many

__all__ = ["Dataset", "File", "ReadPool", "S3Client", "S3Error", "open_s3_many", "read_many"]
