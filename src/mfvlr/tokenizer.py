"""Offline CLIP-compatible tokenizer, approved L02/L03 engineering choices.

The independent BPE implementation uses the pinned official merge data and
cleaning convention. Golden fixtures come from upstream code, not this module.
No learned embedding, model loader, network call or language model is present.
See tokenizer_assets/LICENSE for the upstream MIT attribution.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import gzip
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path

import ftfy
import regex
import torch
from torch import Tensor

ASSET_MANIFEST_SHA256 = "179b071a483620d95f4c16e591bd2e64fd334621c13f76508492569a73d06949"
ASSET_ROOT = Path(__file__).with_name("tokenizer_assets")
BOS, EOS, PAD = 49406, 49407, 0


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def verify_assets(asset_root: Path = ASSET_ROOT) -> dict:
    """Fail closed on modified/missing source, BPE, license or dependency pins."""
    asset_root = Path(asset_root)
    try:
        data = (asset_root / "manifest.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != ASSET_MANIFEST_SHA256:
            raise ValueError("tokenizer manifest SHA-256 mismatch")
        manifest = json.loads(data)
        for name, record in manifest["files"].items():
            blob = (asset_root / name).read_bytes()
            if hashlib.sha256(blob).hexdigest() != record["sha256"]:
                raise ValueError(f"tokenizer asset SHA-256 mismatch: {name}")
            if "decompressed_sha256" in record:
                if hashlib.sha256(gzip.decompress(blob)).hexdigest() != record["decompressed_sha256"]:
                    raise ValueError("decompressed BPE SHA-256 mismatch")
        for name, expected in manifest["dependencies"].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError(f"tokenizer requires {name}=={expected}")
    except (OSError, importlib.metadata.PackageNotFoundError) as error:
        raise ValueError(f"missing tokenizer artifact/dependency: {error}") from error
    return manifest


@dataclass(frozen=True)
class TokenBatch:
    """Per-call record; tensors are caller-owned, not deep immutable objects."""
    ids: Tensor
    segment_ids: Tensor
    payload_lengths: Tensor
    eos_positions: Tensor
    padding_mask: Tensor


class TokenizerAdapter:
    def __init__(self, *, asset_root: Path = ASSET_ROOT) -> None:
        manifest = verify_assets(asset_root)
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        byte_order = visible + [byte for byte in range(256) if byte not in visible]
        symbols = [chr(byte) for byte in visible] + [chr(256+i) for i in range(256-len(visible))]
        self._byte_map = dict(zip(byte_order, symbols))
        lines = gzip.decompress((Path(asset_root) / "clip/bpe_simple_vocab_16e6.txt.gz").read_bytes()).decode("utf-8").split("\n")
        merges = [tuple(line.split()) for line in lines[1:48895]]
        if len(merges) != 48894 or any(len(pair) != 2 for pair in merges):
            raise ValueError("invalid CLIP BPE merge table")
        self._ranks = {pair: index for index, pair in enumerate(merges)}
        vocabulary = symbols + [symbol + "</w>" for symbol in symbols]
        vocabulary += ["".join(pair) for pair in merges] + ["<|startoftext|>", "<|endoftext|>"]
        self._vocab = {symbol: index for index, symbol in enumerate(vocabulary)}
        expected = manifest["vocabulary"]
        if len(self._vocab) != 49408 or _json_hash(vocabulary) != expected["ordered_vocabulary_sha256"]:
            raise ValueError("ordered CLIP vocabulary mismatch")
        if _json_hash(list(map(list, self._byte_map.items()))) != expected["ordered_byte_map_sha256"]:
            raise ValueError("CLIP byte mapping mismatch")
        if self._vocab["<|startoftext|>"] != BOS or self._vocab["<|endoftext|>"] != EOS:
            raise ValueError("CLIP special token IDs mismatch")
        self._pattern = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )

    @classmethod
    def from_config(cls, config):
        config.require_approval()
        return cls()

    def _bpe(self, token: str) -> list[str]:
        if token in ("<|startoftext|>", "<|endoftext|>"):
            return [token]
        parts = list(token[:-1]) + [token[-1] + "</w>"]
        while len(parts) > 1:
            adjacent = list(zip(parts, parts[1:]))
            eligible = [(self._ranks[pair], pair) for pair in adjacent if pair in self._ranks]
            if not eligible:
                break
            _, pair = min(eligible)
            merged = []
            index = 0
            while index < len(parts):
                if index+1 < len(parts) and (parts[index], parts[index+1]) == pair:
                    merged.append(parts[index] + parts[index+1])
                    index += 2
                else:
                    merged.append(parts[index])
                    index += 1
            parts = merged
        return parts

    def encode_text(self, text: str) -> list[int]:
        """Unframed BPE payload; strict 77-slot limit is enforced by encode_levels."""
        if type(text) is not str:
            raise TypeError("text must be a string")
        cleaned = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
        cleaned = regex.sub(r"\s+", " ", cleaned).strip().lower()
        ids = []
        for piece in self._pattern.findall(cleaned):
            encoded = "".join(self._byte_map[byte] for byte in piece.encode("utf-8"))
            ids.extend(self._vocab[part] for part in self._bpe(encoded))
        return ids

    def encode_levels(self, prompts: Sequence[Sequence[str]]) -> TokenBatch:
        if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)):
            raise TypeError("prompts must be a sequence of four-string sequences")
        if not prompts:
            raise ValueError("prompts batch must be positive")
        segments = torch.zeros((len(prompts), 4, 77), dtype=torch.int64, device="cpu")
        lengths = torch.empty((len(prompts), 4), dtype=torch.int64, device="cpu")
        eos_positions = torch.empty_like(lengths)
        padding = torch.ones((len(prompts), 4, 77), dtype=torch.bool, device="cpu")
        for row, levels in enumerate(prompts):
            if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or len(levels) != 4:
                raise ValueError("each image must have exactly four prompt levels")
            for level, text in enumerate(levels):
                payload = self.encode_text(text)
                if len(payload) > 75:
                    raise ValueError(f"image {row} level L{level+1}: {len(payload)} payload tokens exceeds 75; truncation forbidden")
                framed = [BOS] + payload + [EOS]
                segments[row, level, :len(framed)] = torch.tensor(framed, dtype=torch.int64)
                lengths[row, level] = len(payload)
                eos_positions[row, level] = level*77 + len(payload)+1
                padding[row, level, :len(framed)] = False
        return TokenBatch(segments.reshape(len(prompts), 308), segments, lengths,
                          eos_positions, padding.reshape(len(prompts), 308))
