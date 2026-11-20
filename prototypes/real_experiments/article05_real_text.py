"""Train a tiny causal character model on the public Tiny Shakespeare corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F


def prepare_corpus(
    text: str,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
) -> dict:
    if not text:
        raise ValueError("text must not be empty")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    train_end = int(len(text) * train_fraction)
    validation_end = train_end + int(len(text) * validation_fraction)
    pieces = {
        "train": text[:train_end],
        "validation": text[train_end:validation_end],
        "test": text[validation_end:],
    }
    vocabulary = sorted(set(pieces["train"]))
    unseen = (set(pieces["validation"]) | set(pieces["test"])) - set(vocabulary)
    if unseen:
        sample = "".join(sorted(unseen)[:10])
        raise ValueError(f"characters outside training vocabulary: {sample!r}")
    encoder = {character: index for index, character in enumerate(vocabulary)}
    return {
        "encoded": {
            name: np.fromiter((encoder[ch] for ch in chunk), dtype=np.int64, count=len(chunk))
            for name, chunk in pieces.items()
        },
        "split_characters": {name: len(chunk) for name, chunk in pieces.items()},
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
    }


class TinyCharTransformer(nn.Module):
    def __init__(self, vocabulary_size: int, context: int = 64, embedding: int = 32) -> None:
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocabulary_size, embedding)
        self.position_embedding = nn.Embedding(context, embedding)
        self.attention = nn.MultiheadAttention(embedding, num_heads=4, batch_first=True)
        self.norm_attention = nn.LayerNorm(embedding)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding, embedding * 2), nn.GELU(), nn.Linear(embedding * 2, embedding)
        )
        self.norm_feed_forward = nn.LayerNorm(embedding)
        self.output = nn.Linear(embedding, vocabulary_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        attention, _ = self.attention(hidden, hidden, hidden, attn_mask=causal_mask, need_weights=False)
        hidden = self.norm_attention(hidden + attention)
        hidden = self.norm_feed_forward(hidden + self.feed_forward(hidden))
        return self.output(hidden)


def _sample_batch(data: np.ndarray, batch_size: int, context: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - context - 1, (batch_size,))
    windows = np.stack([data[int(start) : int(start) + context + 1] for start in starts])
    batch = torch.as_tensor(windows, dtype=torch.long, device=device)
    return batch[:, :-1], batch[:, 1:]


def _evaluate(model: nn.Module, data: np.ndarray, context: int, batch_size: int, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    starts = list(range(0, len(data) - context - 1, context))
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            block_starts = starts[offset : offset + batch_size]
            windows = np.stack([data[start : start + context + 1] for start in block_starts])
            batch = torch.as_tensor(windows, dtype=torch.long, device=device)
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction="sum")
            total_loss += float(loss)
            total_tokens += batch_size * 0 + int(batch[:, 1:].numel())
    return total_loss / total_tokens


def run_real_experiment(
    data_path: Path,
    seed: int = 3401,
    steps: int = 200,
    batch_size: int = 16,
    context: int = 64,
) -> dict:
    if steps < 1 or batch_size < 1 or context < 2:
        raise ValueError("steps and batch_size must be positive and context must be at least 2")
    data_path = Path(data_path)
    raw_bytes = data_path.read_bytes()
    text = raw_bytes.decode("utf-8")
    corpus = prepare_corpus(text)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    train = corpus["encoded"]["train"]
    validation = corpus["encoded"]["validation"]
    test = corpus["encoded"]["test"]
    model = TinyCharTransformer(corpus["vocabulary_size"], context=context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    unigram_counts = np.bincount(train, minlength=corpus["vocabulary_size"]).astype(np.float64)
    unigram_probabilities = (unigram_counts + 1.0) / (unigram_counts.sum() + len(unigram_counts))
    unigram_test_loss = float(np.mean(-np.log(unigram_probabilities[test])))
    history = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        x, y = _sample_batch(train, batch_size, context, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            validation_loss = _evaluate(model, validation, context, batch_size, device)
            history.append(
                {"step": step, "train_batch_cross_entropy": float(loss.detach()), "validation_cross_entropy": validation_loss}
            )
    train_seconds = time.perf_counter() - started
    test_loss = _evaluate(model, test, context, batch_size, device)
    return {
        "article_id": "05",
        "experiment": "Train a tiny causal character Transformer on public-domain play text on CPU",
        "evidence_type": "measured_public_text",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Tiny Shakespeare subset",
            "source_url": "https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt",
            "source_bytes": len(raw_bytes),
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "characters": len(text),
            "lines": text.count("\n") + 1,
            "vocabulary_size": corpus["vocabulary_size"],
            "split_characters": corpus["split_characters"],
            "split_method": "contiguous first 90% train, next 5% validation, final 5% test; vocabulary fit on train only",
        },
        "method": {
            "architecture": "one-layer causal Transformer, embedding=32, four attention heads, feed-forward width=64",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "context_characters": context,
            "batch_size": batch_size,
            "steps": steps,
            "seed": seed,
            "optimizer": "AdamW(lr=0.002)",
            "baseline": "smoothed train-only unigram character model",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "torch": version("torch"),
            "numpy": version("numpy"),
            "device": "CPU",
            "torch_threads": torch.get_num_threads(),
        },
        "training_wall_seconds": train_seconds,
        "training_characters_processed": steps * batch_size * context,
        "training_characters_per_second": steps * batch_size * context / train_seconds,
        "unigram_test_cross_entropy": unigram_test_loss,
        "transformer_test_cross_entropy": test_loss,
        "unigram_test_perplexity": float(np.exp(unigram_test_loss)),
        "transformer_test_perplexity": float(np.exp(test_loss)),
        "history": history,
        "limitations": [
            "This is character-level prediction on a 1.1 MB subset of Shakespeare, not a modern natural-language model.",
            "A contiguous split means the test excerpt comes from the final portion of the collection; results depend on that ordering.",
            "One seed and one corpus do not establish general language quality or performance on other CPUs.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("evidence/05_real/data/tinyshakespeare.txt"))
    parser.add_argument("--out", type=Path, default=Path("evidence/05_real"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3401)
    args = parser.parse_args(argv)
    result = run_real_experiment(args.data, seed=args.seed, steps=args.steps)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = args.out / f"article-05-tinyshakespeare-{stamp}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(path)
    print("parameters", result["method"]["parameters"])
    print("train_seconds", round(result["training_wall_seconds"], 3))
    print("unigram_test_ce", round(result["unigram_test_cross_entropy"], 5))
    print("transformer_test_ce", round(result["transformer_test_cross_entropy"], 5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
