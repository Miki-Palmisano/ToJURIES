"""Evaluator: define judges, prompt and options once, then evaluate a dataset."""

from __future__ import annotations

import json
from pathlib import Path

from .analysis.pipeline import analyze as _analyze
from .analysis.tables import system_of
from .config import ConfigError, RunConfig
from .dataset import PairDataset
from .prompt import Prompt
from .judje import Provider
from .runner import load_records, run_all


class Run:
    """Result of an evaluation: a folder on disk."""

    def __init__(self, out_dir: Path, counts: dict[str, int]) -> None:
        self.out_dir = out_dir
        self.counts = counts  # {"ok", "cached", "error"}

    def records(self) -> list[dict]:
        return load_records(self.out_dir)

    def analyze(self) -> dict:
        """Write report.md and the CSV tables into the folder; returns the main tables."""
        return _analyze(self.out_dir)

    def __repr__(self) -> str:
        return f"Run(out_dir={str(self.out_dir)!r}, counts={self.counts})"


class Evaluator:
    """
    ``Evaluator(judges=[...], prompt=Prompt.default(), systems=("PRE", "POST"))``

    ``evaluate(dataset, out_dir)`` calls every judge on every pair in every presentation order
    and writes into ``out_dir``:

      config.json   judges (no keys), options, prompt hash
      prompt.md     the prompt used
      runs/<judge>/<pair>__<order>__r<rep>.json   raw reply + parsed JSON, one per call
      results.json  all parsed verdicts, with the system behind REPORT A / REPORT B
    """

    def __init__(self, judges: list[Provider], prompt: Prompt | None = None, **options) -> None:
        if not judges:
            raise ConfigError("at least one judge is required")
        names = [j.name for j in judges]
        if len(set(names)) != len(names):
            raise ConfigError(f"judge names must be unique, got {names}")
        self.judges = list(judges)
        self.prompt = prompt or Prompt.default()
        try:
            self.config = RunConfig(**options)
        except Exception as exc:  # pydantic.ValidationError
            raise ConfigError(f"invalid options: {exc}") from exc

    def evaluate(self, dataset: PairDataset, out_dir: str | Path = "results", *, force: bool = False,
                 limit: int | None = None, analyze: bool = False) -> Run:
        """Run all calls (finished ones are cached in ``out_dir``; failed ones are retried)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "prompt.md").write_text(self.prompt.text, encoding="utf-8")
        (out / "config.json").write_text(json.dumps({
            "options": self.config.model_dump(),
            "judges": [j.describe() for j in self.judges],
            "prompt_hash": self.prompt.hash,
        }, indent=1), encoding="utf-8")
        counts = run_all(dataset, self.judges, self.prompt, self.config, out, force=force, limit=limit)
        self._write_results(out)
        run = Run(out, counts)
        if analyze and run.records():
            run.analyze()
        return run

    def _write_results(self, out: Path) -> None:
        rows = []
        for rec in load_records(out):
            job = rec["job"]
            rows.append({**job,
                         "report_A": system_of("A", job["order"], self.config.systems),
                         "report_B": system_of("B", job["order"], self.config.systems),
                         "verdict": rec["verdict"]})
        (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
