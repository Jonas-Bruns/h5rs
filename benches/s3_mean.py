"""Check transport diagnostics and measure TLS reuse and streaming mean memory."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import ssl
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter, sleep

import h5py
import h5rs
import numpy as np

from viewer_s3 import RangeHandler, Server, endpoint


def make_certificates(root, openssl):
    ca, key, cert, csr, ext = [root / name for name in ("ca.pem", "key.pem", "leaf.pem", "leaf.csr", "leaf.ext")]
    ext.write_text("subjectAltName=IP:127.0.0.1\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n")

    def run(*args):
        subprocess.run([openssl, *args], check=True, capture_output=True)

    root_key = root / "ca.key"
    run("req", "-new", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
        "-subj", "/CN=h5rs test CA", "-keyout", root_key, "-out", ca)
    run("req", "-new", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=localhost",
        "-keyout", key, "-out", csr)
    run("x509", "-req", "-in", csr, "-CA", ca, "-CAkey", root_key, "-CAcreateserial",
        "-days", "2", "-extfile", ext, "-out", cert)
    return ca, cert, key


class HttpsServer(Server):
    def __init__(self, path, cert, key):
        self.handshakes = 0
        self.fault = None
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(cert, key)
        super().__init__([path] * 24, 0)
        self.RequestHandlerClass = TestHandler

    def get_request(self):
        conn, address = super().get_request()
        try:
            if self.fault == "slow_connect":
                sleep(0.35)
            secure = self.context.wrap_socket(conn, server_side=True)
        except OSError:
            conn.close()
            raise
        self.handshakes += 1
        return secure, address


class TestHandler(RangeHandler):
    def do_GET(self):
        fault = self.server.fault
        if isinstance(fault, int):
            self.send_error(fault)
        else:
            super().do_GET()

    def end_headers(self):
        super().end_headers()
        if self.server.fault == "slow_body":
            self.wfile.flush()
            sleep(0.35)

    def handle(self):
        try:
            super().handle()
        except (ConnectionError, ssl.SSLError):
            pass


def expect_error(call, kind, status=None):
    try:
        call()
    except h5rs.S3Error as error:
        assert isinstance(error, OSError)
        assert (error.kind, error.status_code) == (kind, status), (error.kind, error.status_code, str(error))
        assert error.reason and "secret-token" not in str(error)
        return error.reason
    raise AssertionError(f"expected S3 error {kind}")


def benchmark_transport(root, openssl):
    ca, cert, key = make_certificates(root, openssl)
    path = root / "fixture.h5"
    data = np.arange(256 * 128, dtype=np.int32).reshape(256, 128)
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=data, chunks=(16, 32), compression="gzip")
    with HttpsServer(path, cert, key) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host = f"https://127.0.0.1:{server.server_port}"
            url = f"{host}/0?signature=secret-token"
            invalid = root / "invalid.pem"
            invalid.write_text("not a CA")
            assert expect_error(lambda: h5rs.S3Client(ca_bundle=invalid), "ca_bundle")
            errors = {"tls": expect_error(lambda: h5rs.File.open_s3(url), "tls")}
            os.environ["AWS_CA_BUNDLE"] = str(ca)
            np.testing.assert_array_equal(h5rs.File.open_s3(url)["data"][:16, :32], data[:16, :32])
            os.environ["SSL_CERT_FILE"] = str(invalid)
            expect_error(lambda: h5rs.File.open_s3(url), "ca_bundle")
            np.testing.assert_array_equal(h5rs.File.open_s3(url, ca_bundle=ca)["data"][:16, :32], data[:16, :32])
            os.environ["SSL_CERT_FILE"] = str(ca)
            os.environ["AWS_CA_BUNDLE"] = str(invalid)
            np.testing.assert_array_equal(h5rs.File.open_s3(url)["data"][:16, :32], data[:16, :32])
            try:
                h5rs.S3Client(request_timeout=float("nan"))
            except ValueError:
                pass
            else:
                raise AssertionError("invalid timeout accepted")

            pool = h5rs.ReadPool(4, 4, max_s3_gets=8)
            client = h5rs.S3Client(ca_bundle=ca, connect_timeout=2, request_timeout=0.20)
            for mode, settings in (("sync", None), ("async", pool)):
                dataset = h5rs.File.open_s3(url, settings, client=client).dataset_with_s3_index("data")
                np.testing.assert_array_equal(dataset[:32, :64], data[:32, :64])
                for status in (404, 503):
                    server.fault = status
                    errors[f"{mode}_{status}"] = expect_error(lambda: dataset[:32, :64], "http", status)
                server.fault = "slow_body"
                errors[f"{mode}_body"] = expect_error(lambda: dataset[:32, :64], "timeout")
                server.fault = None

            client = h5rs.S3Client(ca_bundle=ca, connect_timeout=0.10, request_timeout=2)
            dataset = h5rs.File.open_s3(url, pool, client=client).dataset_with_s3_index("data")
            server.fault = "slow_connect"
            try:
                errors["async_connect_timeout"] = expect_error(lambda: dataset[:16, :32], "timeout")
                errors["sync_connect_timeout"] = expect_error(
                    lambda: h5rs.File.open_s3(url, ca_bundle=ca, connect_timeout=0.10, request_timeout=2),
                    "timeout")
            finally:
                server.fault = None
            sleep(0.40)

            results = {}
            for label in ("separate_clients", "shared_client"):
                shared = h5rs.S3Client(ca_bundle=ca) if label == "shared_client" else None
                before = server.handshakes
                opens = []
                tiles = []
                for index in range(24):
                    started = perf_counter()
                    file = h5rs.File.open_s3(f"{host}/{index}", client=shared, ca_bundle=None if shared else ca)
                    opens.append((perf_counter() - started) * 1000)
                    file["data"][:32, :64]
                    tiles.append((perf_counter() - started) * 1000)
                results[label] = {
                    "opens": len(opens), "handshakes": server.handshakes - before,
                    "median_open_ms": median(opens), "median_first_tile_ms": median(tiles),
                }
            assert results["shared_client"]["handshakes"] < results["separate_clients"]["handshakes"]
        finally:
            for variable in ("SSL_CERT_FILE", "AWS_CA_BUNDLE"):
                os.environ.pop(variable, None)
            server.shutdown()
            thread.join()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    errors["connect"] = expect_error(lambda: h5rs.File.open_s3(f"http://127.0.0.1:{port}/"), "connect")
    return {"open_benchmark": results, "errors": errors}


def peak_mib():
    if sys.platform != "win32":
        import resource
        divisor = 1024**2 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor

    class Counters(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in ("peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.K32GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak / 1024**2


def mean_worker(path, mode):
    if path.startswith("http://"):
        dataset = h5rs.File.open_s3(path).dataset_with_s3_index("data")
    else:
        dataset = h5rs.File(path)["data"]
    rows, cols = dataset.shape
    selection = (slice(None), slice(None))
    samples = []
    base_peak = peak_mib()
    for _ in range(3):
        start = perf_counter()
        if mode == "numpy":
            result = dataset[selection].reshape(-1, 64, cols).mean(axis=1, dtype=np.float64)
        else:
            result = dataset.read_mean(selection, block=64)
        samples.append((perf_counter() - start) * 1000)
        checksum = float(result.sum())
        del result
    return {"median_ms": median(samples), "peak_mib": peak_mib(), "baseline_peak_mib": base_peak,
            "checksum": checksum, "source_mib": rows * cols * 4 / 1024**2, "output_mib": rows * cols / 64 * 8 / 1024**2}


def benchmark_mean(root):
    path = root / "mean.h5"
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("data", (32768, 1024), dtype="f4", chunks=(128, 256),
                                      compression="gzip", shuffle=True, compression_opts=1)
        for start in range(0, 32768, 512):
            dataset[start:start + 512] = rng.standard_normal((512, 1024), dtype=np.float32)
    def compare(source, server=None):
        results = {}
        # Separate processes keep peak memory measurements independent.
        for mode in ("numpy", "rust"):
            if server:
                server.reset()
            completed = subprocess.run([sys.executable, __file__, "--worker", mode, str(source)],
                                       text=True, capture_output=True, check=True)
            results[mode] = json.loads(completed.stdout)
            if server:
                results[mode]["range_gets"] = server.stats["gets"]
                results[mode]["range_bytes"] = server.stats["bytes"]
        np.testing.assert_allclose(results["numpy"]["checksum"], results["rust"]["checksum"])
        return results

    local = compare(path)
    with endpoint([path], 0.0005) as (server, urls):
        remote = compare(urls[0], server)
    return {"local": local, "remote": remote}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--openssl", default="openssl")
    parser.add_argument("--worker", choices=("numpy", "rust"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(mean_worker(args.output, args.worker)))
        return
    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        environment = {name: os.environ.pop(name, None) for name in ("SSL_CERT_FILE", "AWS_CA_BUNDLE")}
        try:
            s3 = benchmark_transport(root, args.openssl)
        finally:
            os.environ.update({name: value for name, value in environment.items() if value is not None})
        results = {"s3": s3, "mean": benchmark_mean(root)}
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
