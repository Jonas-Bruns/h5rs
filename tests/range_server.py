"""Small HTTP/1.1 fixture server with S3-style conditional byte ranges."""

import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer


ROOT = Path(sys.argv[1])
LOCK = threading.Lock()
active = peak = reads = read_bytes = 0


class RangeServer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        global active, peak, reads, read_bytes
        mode = self.path.split("?", 1)[0]
        fixture = ROOT / {
            "/latest": "latest.h5",
            "/large-index": "large-index.h5",
        }.get(mode, "reference.h5")
        size = fixture.stat().st_size
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not match:
            self.send_error(400)
            return
        start, end = map(int, match.groups())
        if not start <= end < size:
            self.send_error(416)
            return
        if mode != "/latest" and end - start + 1 == size:
            self.send_error(400, "unexpected whole-object transfer")
            return
        etag = '"fixture-version-1"'
        if mode == "/changed" and (ROOT / "changed").exists():
            etag = '"fixture-version-2"'
        condition = self.headers.get("If-Match")
        if condition != etag and not (condition is None and start == end == 0):
            self.send_error(412)
            return
        if mode == "/bad-metadata" and end > start:
            self.send_error(503)
            return
        length = end - start + 1
        if mode == "/short":
            length -= 1
        if mode == "/weak":
            etag = "W/" + etag
        is_chunk = 1 < length < 1024 and start > 128 * 1024
        if is_chunk:
            with LOCK:
                active += 1
                peak = max(active, peak)
                reads += 1
                read_bytes += length
                (ROOT / "chunk-reads").write_text(f"{peak} {reads} {read_bytes}", encoding="utf-8")
        try:
            if is_chunk:
                time.sleep(0.025)
            self.send_response(200 if mode == "/ignored" else 206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("ETag", etag)
            self.end_headers()
            with fixture.open("rb") as file:
                file.seek(start)
                self.wfile.write(file.read(length))
        finally:
            if is_chunk:
                with LOCK:
                    active -= 1


class FixtureServer(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer normally looks up the bound address with socket.getfqdn().
        # This loopback fixture can announce its port without waiting for DNS.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


if __name__ == "__main__":
    with FixtureServer(("127.0.0.1", 0), RangeServer) as server:
        (ROOT / "port").write_text(str(server.server_port), encoding="utf-8")
        server.serve_forever()
