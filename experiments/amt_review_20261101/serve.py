"""Serve only report-manifest entries; never expose research roots or directory listings."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
from http.client import HTTPConnection

ROOT = Path(__file__).resolve().parents[3] / 'audit/revision-review-20261101'
PUBLIC = ROOT / 'public'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def proxy_review(self):
        path=urlsplit(self.path).path
        port=8770 if path.startswith('/experimental-review/') else 8771 if path.startswith('/experimental-monitor/') else None
        if port is None:
            return False
        if port==8771 and self.command!='GET':
            self.send_error(405)
            return True
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.send_error(400)
            return True
        if not 0 <= size <= 16384:
            self.send_error(413)
            return True
        body = self.rfile.read(size) if size else None
        headers = {k: self.headers[k] for k in ['Content-Type', 'Cookie', 'X-CSRF-Token'] if self.headers.get(k)}
        connection = HTTPConnection('127.0.0.1', port, timeout=30)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {'connection', 'transfer-encoding', 'server', 'date', 'content-length'}:
                    self.send_header(key, value)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, TimeoutError):
            self.send_error(503, 'Experimental review service is unavailable')
        finally:
            connection.close()
        return True

    def do_GET(self):
        if not self.proxy_review():
            super().do_GET()

    def do_POST(self):
        if not self.proxy_review():
            self.send_error(404)

    def send_head(self):
        path = unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
        allowlist = json.loads((ROOT / 'public-manifest.json').read_text())['files']
        resolved = (PUBLIC / path).resolve()
        if path not in allowlist or not resolved.is_relative_to(PUBLIC.resolve()) or not resolved.is_file():
            self.send_error(404)
            return None
        self.path = '/' + path
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Robots-Tag', 'noindex, nofollow')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        super().end_headers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8769)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Report server: 127.0.0.1:{args.port}; root={PUBLIC}', flush=True)
    server.serve_forever()
