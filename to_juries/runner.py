"""Run every (pair x judge x order x repetition) call, cached on disk."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import hashlib

from .config import ConfigError, RunConfig
from .dataset import Pair
from .parsing import ParseError, extract_verdict
from .prompt import Prompt
from .judje import Provider

log = logging.getLogger("reportjury")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


@dataclass(frozen=True)
class Job:
    pair: Pair
    judge: Provider
    prompt: Prompt
    order: str  # "AB": A = first system, B = second ; "BA": reversed
    rep: int

    def path(self, out_dir: Path) -> Path:
        return out_dir / "runs" / _safe(self.judge.name) / f"{_safe(self.pair.pair_id)}__{self.order}__r{self.rep}.json"

    def rendered(self) -> str:
        a, b = self.pair.baseline_text, self.pair.candidate_text
        if self.order == "BA":
            a, b = b, a
        return self.prompt.render(task_prompt=self.pair.task_prompt, reference=self.pair.reference_text,
                                  cutoff=self.pair.cutoff, report_a=a, report_b=b)

    def hash(self, rendered: str) -> str:
        h = hashlib.sha256()
        for t in (json.dumps(self.judge.describe(), sort_keys=True), rendered):
            h.update(t.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]


def build_jobs(pairs, judges: list[Provider], prompt: Prompt, cfg: RunConfig) -> list[Job]:
    return [Job(p, j, prompt, o, r) for p in pairs for j in judges for o in cfg.orders for r in range(cfg.reps)]


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def run_job(job: Job, cfg: RunConfig, out_dir: Path, force: bool) -> str:
    """Returns "cached", "ok" or "error"."""
    path = job.path(out_dir)
    prompt = job.rendered()
    h = job.hash(prompt)
    if path.exists() and not force:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("prompt_hash") == h and not old.get("error"):
                return "cached"
        except json.JSONDecodeError:
            pass

    record = {"job": {"pair_id": job.pair.pair_id, "judge": job.judge.name,
                      "order": job.order, "rep": job.rep}, "prompt_hash": h}
    note, last = "", None
    for attempt in range(cfg.max_retries):
        try:
            resp = job.judge.complete(prompt + note)
            verdict = extract_verdict(resp.text)
            record.update(raw_text=resp.text, verdict=verdict.model_dump(), usage=resp.usage, error=None)
            _write(path, record)
            return "ok"
        except ConfigError:
            raise
        except ParseError as exc:
            last = exc
            note = ("\n\nYour previous reply could not be used: "
                    f"{str(exc)[:300]}\nRepeat the evaluation and end with exactly the requested JSON object.")
        except Exception as exc:  # noqa: BLE001 - network, rate limits, empty replies
            last = exc
            time.sleep(min(60.0, 2.0 * 2**attempt))
    record.update(error=f"failed after {cfg.max_retries} attempts: {last}")
    _write(path, record)
    return "error"


def run_all(pairs, judges: list[Provider], prompt: Prompt, cfg: RunConfig, out_dir: str | Path, *,
            force: bool = False, limit: int | None = None) -> dict[str, int]:
    out_dir = Path(out_dir)
    jobs = build_jobs(pairs, judges, prompt, cfg)[:limit]
    counts = {"ok": 0, "cached": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futs = {pool.submit(run_job, j, cfg, out_dir, force): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            status = fut.result()  # ConfigError propagates and aborts the run
            counts[status] += 1
            if status == "error":
                log.warning("job failed: %s", futs[fut].path(out_dir).name)
            if i % 10 == 0 or i == len(jobs):
                log.info("progress %d/%d %s", i, len(jobs), counts)
    return counts


def load_records(out_dir: str | Path) -> list[dict]:
    recs = []
    for p in sorted((Path(out_dir) / "runs").glob("*/*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not r.get("error") and "verdict" in r:
            recs.append(r)
    return recs
