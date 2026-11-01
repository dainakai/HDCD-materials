"""Check every local HTML resource and anchor before exposing this report."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json

from prepare import OUT, sha


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids, f'Duplicate id: {a["id"]}'
            self.ids.add(a['id'])
        for key in ['src', 'href']:
            if key in a:
                self.links.append(a[key])


def main():
    root = OUT / 'public'
    files = json.loads((OUT / 'public-manifest.json').read_text())['files']
    assert not any(p.is_symlink() for p in root.rglob('*'))
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == set(files)
    for name, expected in files.items():
        assert sha(root / name) == expected, name
    parsed = {}
    for name in files:
        if name.endswith('.html'):
            parser = Parser()
            parser.feed((root / name).read_text())
            parsed[name] = parser
    checked = 0
    for name, parser in parsed.items():
        for link in parser.links:
            parts = urlsplit(link)
            if parts.scheme:
                assert parts.scheme in {'https', 'http'}, link
                continue
            target = (root / name).parent / unquote(parts.path) if parts.path else root / name
            target = target.resolve()
            assert target.is_relative_to(root.resolve()), (name, link)
            rel = str(target.relative_to(root.resolve()))
            assert rel in files, (name, link)
            if parts.fragment and rel in parsed:
                assert parts.fragment in parsed[rel].ids, (name, link)
            checked += 1
    result = {'status': 'PASS', 'public_files': len(files), 'html_pages': len(parsed), 'links_and_resources_checked': checked}
    (OUT / 'html-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
