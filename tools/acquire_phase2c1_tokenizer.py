"""Acquire only pinned official tokenizer/reference files; never model weights.

Network is used only by this explicit preparation utility, never at runtime.
"""
from pathlib import Path
import gzip
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
ASSETS = ROOT / "src/mfvlr/tokenizer_assets"
FILES = ("clip/simple_tokenizer.py", "clip/clip.py", "clip/bpe_simple_vocab_16e6.txt.gz", "LICENSE")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/openai/CLIP/{COMMIT}/{name}"
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read()
        destination = ASSETS / name
        if destination.exists() and destination.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite changed pinned asset: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        records[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if name.endswith(".gz"):
            records[name]["decompressed_sha256"] = hashlib.sha256(gzip.decompress(data)).hexdigest()
        print(name, records[name]["sha256"], len(data))
    manifest = {
        "schema_version": 1, "repository": "https://github.com/openai/CLIP",
        "commit": COMMIT, "commit_date": "2026-03-25T18:46:40Z",
        "client_acquisition_date": "2026-10-11", "files": records,
        "learned_model_weights_downloaded": False,
        "source_role": "Official tokenizer and tokenize-function reference; clip.py is never imported wholesale",
        "dependencies": {"ftfy": "6.3.1", "regex": "2026.6.28", "wcwidth": "0.2.13"},
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
