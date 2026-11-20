"""Run the existing CPU smoke test in an isolated temporary copy."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def validate_steps(steps: int) -> int:
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 1000:
        raise ValueError("steps must be a positive integer no greater than 1000")
    return steps


def run(seed: int = 42, steps: int = 20) -> dict:
    validate_steps(steps)
    source = Path(__file__).resolve().parents[2] / "evidence" / "02_05" / "proof_binary_and_cpu.py"
    if not source.is_file():
        raise FileNotFoundError(f"Existing CPU prototype is missing: {source}")
    with tempfile.TemporaryDirectory(prefix="hiai-article05-") as temporary:
        temp_root = Path(temporary)
        staged = temp_root / "evidence" / "02_05" / source.name
        staged.parent.mkdir(parents=True)
        shutil.copy2(source, staged)
        completed = subprocess.run(
            [sys.executable, str(staged), "--steps", str(steps)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        report_path = temp_root / "evidence" / "results" / "cpu_toy_smoke.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "evidence_type": "synthetic",
        "seed": seed,
        "training": {
            "torch_version": report["torch_version"],
            "model_seed": report["seed"],
            "parameters": report["params"],
            "steps": report["steps"],
            "tokens_processed": report["training_tokens_processed"],
            "wall_seconds": report["training_wall_seconds"],
            "tokens_per_second": report["train_tokens_per_second"],
            "test_cross_entropy": report["transformer_test_cross_entropy"],
        },
        "stdout": completed.stdout.strip().splitlines()[-1:] or [],
        "limitations": [
            "The training data is an artificial character grammar, not natural language.",
            "This small CPU smoke test does not establish useful LLM capability.",
            "The wrapper seed labels this run; the wrapped training script uses its own fixed model seed, recorded above.",
        ],
    }
