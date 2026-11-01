"""Serve only report-manifest entries; never expose research roots or directory listings."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3] / 'audit/revision-review-20261101'
PUBLIC = ROOT / 'public'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

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
