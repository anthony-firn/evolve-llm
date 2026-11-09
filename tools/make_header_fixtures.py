"""Tokenizer and chat-template fixtures for models too large to download for tests.

The vocabulary, merges and chat template sit in a GGUF file's header, so only the header of
the first part is fetched (with an HTTP Range request) and saved as <models>/header-<key>.gguf.
Reference token ids come from the model's own Hugging Face tokenizer and reference prompts
from transformers' rendering of the template stored in the GGUF.

Usage: python3 tools/make_header_fixtures.py --models ~/.cache/evolve-llm/models --tokenizers DIR
Needs transformers and tokenizers. No code from the model repositories is run: DeepSeek's
tokenizer.json is read by the tokenizers library, and Kimi K2's tiktoken ranks are read with this
script's own copy of its split pattern and special tokens.
"""
import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fixtures import CONVERSATION, OUT, TEXTS  # noqa: E402

HF = "https://huggingface.co"
MODELS = {
    "deepseek31": ("unsloth/DeepSeek-V3.1-GGUF", "UD-IQ1_S/DeepSeek-V3.1-UD-IQ1_S-00001-of-00004.gguf",
                   "deepseek-ai/DeepSeek-V3.1", ["tokenizer.json", "tokenizer_config.json"]),
}


def fetch(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def header_length(b):
    """Bytes up to the end of the tensor directory, or None if `b` is too short."""
    pos = 4 + 4
    try:
        nt, nk = struct.unpack_from("<QQ", b, pos)
        pos += 16
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def value(t):
            nonlocal pos
            if t in sizes:
                pos += sizes[t]
            elif t == 8:
                (n,) = struct.unpack_from("<Q", b, pos)
                pos += 8 + n
            elif t == 9:
                et, n = struct.unpack_from("<IQ", b, pos)
                pos += 12
                if et in sizes:
                    pos += sizes[et] * n
                else:
                    for _ in range(n):
                        value(et)
            if pos > len(b):
                raise IndexError

        for _ in range(nk):
            (n,) = struct.unpack_from("<Q", b, pos)
            pos += 8 + n
            (t,) = struct.unpack_from("<I", b, pos)
            pos += 4
            value(t)
        for _ in range(nt):
            (n,) = struct.unpack_from("<Q", b, pos)
            pos += 8 + n
            (nd,) = struct.unpack_from("<I", b, pos)
            pos += 4 + 8 * nd + 4 + 8
        return pos if pos <= len(b) else None
    except (struct.error, IndexError):
        return None


def template_of(b):
    """The tokenizer.chat_template string of a GGUF header, or None."""
    key = b"tokenizer.chat_template"
    at = b.find(key)
    if at < 0:
        return None
    (t,) = struct.unpack_from("<I", b, at + len(key))
    (n,) = struct.unpack_from("<Q", b, at + len(key) + 4)
    start = at + len(key) + 12
    return b[start:start + n].decode() if t == 8 else None


def save_header(url, path):
    size = 8 << 20
    while True:
        b = fetch(url, 0, size)
        n = header_length(b)
        if n:
            path.write_bytes(b[:n])
            return n
        size *= 2


# Kimi K2 (moonshotai/Kimi-K2-Instruct-0905): its tokenizer is tiktoken byte-level BPE over these
# ranks, with this split pattern (copied from its tokenization_kimi.py) and its special tokens
# recognised anywhere in the text.
KIMI = ("unsloth/Kimi-K2-Instruct-0905-GGUF", "UD-TQ1_0/Kimi-K2-Instruct-0905-UD-TQ1_0-00001-of-00006.gguf",
        "moonshotai/Kimi-K2-Instruct-0905", ["tiktoken.model", "tokenizer_config.json"])
KIMI_PATTERN = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


class KimiReference:
    def __init__(self, tdir):
        import base64
        import regex
        self.ranks = {}
        for line in (tdir / "tiktoken.model").read_text().splitlines():
            if line.strip():
                tok, rank = line.split()
                self.ranks[base64.b64decode(tok)] = int(rank)
        cfg = json.loads((tdir / "tokenizer_config.json").read_text())
        self.special = {v["content"]: int(k) for k, v in cfg["added_tokens_decoder"].items()}
        self.split = regex.compile(KIMI_PATTERN, flags=regex.V1)
        self.special_re = regex.compile("|".join(regex.escape(t) for t in sorted(self.special, key=len, reverse=True)))

    def bpe(self, piece):
        if piece in self.ranks:
            return [self.ranks[piece]]
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            best = min(range(len(parts) - 1), key=lambda i: self.ranks.get(parts[i] + parts[i + 1], 1 << 60))
            if parts[best] + parts[best + 1] not in self.ranks:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return [self.ranks[p] for p in parts]

    def encode(self, text):
        out, last = [], 0
        for m in list(self.special_re.finditer(text)) + [None]:
            end = m.start() if m else len(text)
            for piece in self.split.findall(text[last:end]):
                out += self.bpe(piece.encode())
            if m:
                out.append(self.special[m.group()])
                last = m.end()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--tokenizers", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    for key, (repo, file, tok_repo, tok_files) in MODELS.items():
        out = a.models / f"header-{key}.gguf"
        if not out.exists():
            print(key, "header", save_header(f"{HF}/{repo}/resolve/main/{file}", out), "bytes")
        tdir = a.tokenizers / key
        tdir.mkdir(parents=True, exist_ok=True)
        for f in tok_files:
            if not (tdir / f).exists():
                urllib.request.urlretrieve(f"{HF}/{tok_repo}/resolve/main/{f}", tdir / f)
        hf = AutoTokenizer.from_pretrained(str(tdir), trust_remote_code=False)
        fx = {"file": out.name, "tokenizer": [{"text": t, "ids": hf.encode(t, add_special_tokens=False)} for t in TEXTS]}
        template = template_of(out.read_bytes())
        if template:
            text = hf.apply_chat_template(CONVERSATION, chat_template=template, tokenize=False, add_generation_prompt=True)
            fx["chat"] = [{"messages": CONVERSATION, "options": {"date": [2026, 11, 8]}, "text": text}]
        (OUT / f"header-{key}.json").write_text(json.dumps(fx, ensure_ascii=False, indent=1))
        print(f"header-{key}.json")
    from transformers.utils.chat_template_utils import render_jinja_template
    repo, file, tok_repo, tok_files = KIMI
    out = a.models / "header-kimik2.gguf"
    if not out.exists():
        print("kimik2 header", save_header(f"{HF}/{repo}/resolve/main/{file}", out), "bytes")
    tdir = a.tokenizers / "kimik2"
    tdir.mkdir(parents=True, exist_ok=True)
    for f in tok_files:
        if not (tdir / f).exists():
            urllib.request.urlretrieve(f"{HF}/{tok_repo}/resolve/main/{f}", tdir / f)
    ref = KimiReference(tdir)
    fx = {"file": out.name, "tokenizer": [{"text": t, "ids": ref.encode(t)} for t in TEXTS]}
    template = template_of(out.read_bytes())
    rendered, _ = render_jinja_template(conversations=[CONVERSATION], chat_template=template, add_generation_prompt=True,
                                        bos_token="[BOS]", eos_token="[EOS]")
    fx["chat"] = [{"messages": CONVERSATION, "options": {"date": [2026, 11, 8]}, "text": rendered[0]}]
    (OUT / "header-kimik2.json").write_text(json.dumps(fx, ensure_ascii=False, indent=1))
    print("header-kimik2.json")


if __name__ == "__main__":
    main()
