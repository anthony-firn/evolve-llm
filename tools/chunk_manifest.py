#!/usr/bin/env python3
"""Chunk-hash manifests that let browsers check model pieces they receive from each other.

A GGUF file is split into 256 KiB chunks. The manifest of a file is:

  v1/<file sha256>/index.json   {"format": "evolve-llm-chunks/1", "sha256", "size", "chunk",
                                 "page_chunks", "pages": [sha256 of each page], "source"}
  v1/<file sha256>/<n>.bin      page n: the 32-byte SHA-256 of chunks n*4096 .. n*4096+4095

Each page covers 1 GiB of the file. A browser trusts index.json (fetched over TLS from the
manifests repository), checks each page it fetches against the index, and each chunk a peer sends
against its page. Files are named by their SHA-256, which Hugging Face lists for every LFS file,
so any copy of the same file in any repository shares one manifest and one swarm.

usage: chunk_manifest.py SOURCE --out DIR [--sha256 EXPECTED] [--repo R --file F --revision C]
SOURCE is a local path or an http(s) URL (a Hugging Face resolve URL works). Only the Python
standard library is used, so it runs anywhere, including CI runners with little disk: the file is
streamed and never stored.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

CHUNK = 256 * 1024
PAGE_CHUNKS = 4096
READ = 8 << 20


def open_source(src):
    if src.startswith(('http://', 'https://')):
        req = urllib.request.Request(src, headers={'User-Agent': 'evolve-llm-manifest'})
        resp = urllib.request.urlopen(req, timeout=120)
        return resp, int(resp.headers.get('Content-Length') or 0)
    return open(src, 'rb'), os.path.getsize(src)


def build(src, out_root, expect=None, source=None, quiet=False):
    stream, size = open_source(src)
    whole = hashlib.sha256()
    page = bytearray()
    page_hashes = []
    pending = bytearray()
    total = 0
    t0 = time.time()
    tmp = os.path.join(out_root, 'v1', '.partial')
    os.makedirs(tmp, exist_ok=True)

    def flush_page():
        name = os.path.join(tmp, f'{len(page_hashes)}.bin')
        with open(name, 'wb') as f:
            f.write(page)
        page_hashes.append(hashlib.sha256(page).hexdigest())
        page.clear()

    def add_chunk(data):
        page.extend(hashlib.sha256(data).digest())
        if len(page) == PAGE_CHUNKS * 32:
            flush_page()

    with stream:
        while True:
            data = stream.read(READ)
            if not data:
                break
            whole.update(data)
            total += len(data)
            pending.extend(data)
            n = len(pending) // CHUNK * CHUNK
            for i in range(0, n, CHUNK):
                add_chunk(bytes(pending[i:i + CHUNK]))
            del pending[:n]
            if not quiet and total % (1 << 30) < READ:
                rate = total / max(1e-9, time.time() - t0) / 1e6
                print(f'{total / 1e9:.1f} GB hashed, {rate:.0f} MB/s', file=sys.stderr, flush=True)
    if pending:
        add_chunk(bytes(pending))
    if page:
        flush_page()
    if size and total != size:
        raise SystemExit(f'read {total} bytes, expected {size}')
    sha = whole.hexdigest()
    if expect and sha != expect.lower():
        raise SystemExit(f'the file hashes to {sha}, not the expected {expect}')
    final = os.path.join(out_root, 'v1', sha)
    if os.path.exists(final):
        for f in os.listdir(final):
            os.remove(os.path.join(final, f))
        os.rmdir(final)
    os.rename(tmp, final)
    index = {
        'format': 'evolve-llm-chunks/1',
        'sha256': sha,
        'size': total,
        'chunk': CHUNK,
        'page_chunks': PAGE_CHUNKS,
        'pages': page_hashes,
        'source': source or {},
    }
    with open(os.path.join(final, 'index.json'), 'w') as f:
        json.dump(index, f, indent=1)
        f.write('\n')
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sha256', help='expected SHA-256 of the whole file (Hugging Face lists it as the LFS oid)')
    ap.add_argument('--repo')
    ap.add_argument('--file')
    ap.add_argument('--revision')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    source = {k: v for k, v in (('repo', a.repo), ('file', a.file), ('revision', a.revision)) if v}
    index = build(a.source, a.out, a.sha256, source, a.quiet)
    print(json.dumps({'sha256': index['sha256'], 'size': index['size'], 'pages': len(index['pages'])}))


if __name__ == '__main__':
    main()
