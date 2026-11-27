"""Run every (pair x judge x order x repetition) call, cached on disk."""

# Enables modern type annotations (e.g. `str | Path`, `dict[str, int]`) on older Python versions.
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, RunConfig
from .dataset import Pair
from .parsing import ParseError, extract_verdict
from .prompt import Prompt
from .judje import Provider

# Package logger: the caller decides level and output (e.g. logging.basicConfig(level=logging.INFO)).
log = logging.getLogger("to_juries")


# =============================================================================
# 1. FILE NAMES
# =============================================================================

def _safe(name: str) -> str:
    """
    Make a string usable as part of a file name, replacing every character other than
    letters, digits, ".", "_" and "-" with "_".
    :param name: judge name or pair_id
    :return: the cleaned string
    """
    # Each run of disallowed characters (spaces, slashes, colons, ...) becomes a single "_".
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# =============================================================================
# 2. ONE CALL
# =============================================================================

# frozen=True: a job is immutable once created.
@dataclass(frozen=True)
class Job:
    pair: Pair # the pair to evaluate
    judge: Provider # the judge making the call
    prompt: Prompt # the evaluation prompt
    order: str # "AB": A = first system, B = second ; "BA": reversed
    rep: int # repetition number (0, 1, ...)

    def path(
            self,
            out_dir: Path
    ) -> Path:
        """
        Path of the JSON file where this call is saved: ``runs/<judge>/<pair>__<order>__r<rep>.json``.
        :param out_dir: results folder
        :return: the path of the file (it may not exist yet)
        """
        # One folder per judge, one file per (pair, order, repetition).
        # The file name does not contain the cache key: a new call with a changed
        # prompt or judge overwrites the old file.
        return out_dir / "runs" / _safe(self.judge.name) / f"{_safe(self.pair.pair_id)}__{self.order}__r{self.rep}.json"

    def rendered(self) -> str:
        """
        Build the prompt of this call. With order "AB" the first report is shown as REPORT A and the
        second as REPORT B; with "BA" they are swapped.
        :return: the prompt with the five placeholders filled
        """
        a, b = self.pair.first_text, self.pair.second_text

        # Swapped order, to control the judges' position bias.
        if self.order == "BA":
            a, b = b, a
        return self.prompt.render(task_prompt=self.pair.task_prompt, reference=self.pair.reference_text,
                                  cutoff=self.pair.cutoff, report_a=a, report_b=b)

    def hash(
            self,
            rendered: str
    ) -> str:
        """
        Cache key of this call: a fingerprint of the judge configuration (model, temperature, ...)
        and of the rendered prompt. If either changes, the saved result is not reused.
        :param rendered: the prompt returned by ``rendered()``
        :return: the first 16 characters of the SHA-256
        """
        h = hashlib.sha256()

        # sort_keys=True: the same configuration always gives the same JSON string.
        # The "\x00" separator keeps the two parts distinct (no ambiguity at the boundary).
        for t in (json.dumps(self.judge.describe(), sort_keys=True), rendered):
            h.update(t.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]


# =============================================================================
# 3. LIST OF CALLS
# =============================================================================

def build_jobs(
        pairs,
        judges: list[Provider],
        prompt: Prompt,
        cfg: RunConfig
) -> list[Job]:
    """
    List all the calls to make: one for each combination of pair, judge, order and repetition.
    :param pairs: the pairs to evaluate
    :param judges: the judges
    :param prompt: the evaluation prompt
    :param cfg: run options (``orders`` and ``reps`` decide how many calls per pair and judge)
    :return: the list of jobs, ordered by pair, judge, order and repetition
    """
    # Cartesian product: n_pairs x n_judges x n_orders x reps calls in total.
    return [Job(p, j, prompt, o, r) for p in pairs for j in judges for o in cfg.orders for r in range(cfg.reps)]


def _check_paths(
        jobs: list[Job],
        out_dir: Path
) -> None:
    """
    Make sure no two calls would be saved to the same file. This can happen because ``_safe``
    maps different names to the same one (e.g. "a/b" and "a b" both become "a_b"), or because the
    file system ignores case (Windows, macOS).
    :param jobs: the calls of the run
    :param out_dir: results folder
    :raises ConfigError: if two calls share a file, naming both
    """
    # Comparison is case-insensitive, to be safe on every file system.
    seen: dict[str, Job] = {}
    for j in jobs:
        key = str(j.path(out_dir)).lower()
        other = seen.setdefault(key, j)
        if other is not j:
            raise ConfigError(
                f"two calls would be saved to the same file {j.path(out_dir).relative_to(out_dir)}: "
                f"judge {other.judge.name!r} / pair {other.pair.pair_id!r} and "
                f"judge {j.judge.name!r} / pair {j.pair.pair_id!r}; rename one of them")


# =============================================================================
# 4. SAFE WRITING
# =============================================================================

def _write(
        path: Path,
        data: dict
) -> None:
    """
    Save a JSON file safely: it is written to a temporary file and then renamed, so an interrupted
    run never leaves a half-written result.
    :param path: destination file (the folder is created if needed)
    :param data: content to save
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file next to the destination (same folder -> same file system,
    # so the rename is atomic).
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # os.replace overwrites the destination atomically, on every OS.
    os.replace(tmp, path)


# =============================================================================
# 5. RUNNING ONE CALL (WITH CACHE AND RETRIES)
# =============================================================================

def run_job(
        job: Job,
        cfg: RunConfig,
        out_dir: Path,
        force: bool
) -> str:
    """
    Execute one call and save its JSON. If a valid result with the same cache key already exists
    it is reused. Otherwise the judge is called, up to ``max_retries`` times: when the reply cannot
    be parsed the prompt is repeated with a note on the error; on network or rate-limit errors it
    waits (about 2, 4, 8... seconds, at most 60, with random jitter) before retrying. After the last
    attempt the error is saved in the file, together with the last reply received (if any).
    :param job: the call to make
    :param cfg: run options (``max_retries``)
    :param out_dir: results folder
    :param force: if True, ignore the cache
    :return: "cached" (reused), "ok" (new valid result) or "error" (failed after all the attempts)
    :raises ConfigError: for configuration problems (e.g. a missing key): the whole run is aborted
    :raises OSError: if the result cannot be written to disk
    """
    path = job.path(out_dir)
    prompt = job.rendered()
    h = job.hash(prompt)

    # --- Cache ---------------------------------------------------------------
    # Reuse the saved result only if it has the same cache key and is not a failed call.
    # A corrupted or unreadable file is treated as missing: the call is made again.
    if path.exists() and not force:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and old.get("cache_key") == h and not old.get("error"):
                return "cached"
        except (OSError, ValueError):
            pass

    # --- Record to save ------------------------------------------------------
    # Job metadata + cache key (judge configuration + rendered prompt).
    record = {"job": {"pair_id": job.pair.pair_id, "judge": job.judge.name,
        "order": job.order, "rep": job.rep}, "cache_key": h}

    # --- Attempts ------------------------------------------------------------
    # note: text appended to the prompt after an unparsable reply; last: last error seen;
    # last_text: last reply received (FIX: saved with the error, for debugging).
    note, last, last_text = "", None, None
    for attempt in range(cfg.max_retries):
        is_last = attempt == cfg.max_retries - 1
        try:
            resp = job.judge.complete(prompt + note)
            last_text = resp.text
            verdict = extract_verdict(resp.text)
        except ConfigError:
            # Configuration problems (e.g. missing API key) would fail on every call:
            # propagate immediately instead of retrying.
            raise
        except ParseError as exc:
            # The model replied but the final JSON is unusable: retry at once,
            # telling the model what went wrong (error message capped at 300 characters).
            last = exc
            note = ("\n\nYour previous reply could not be used: "
                    f"{str(exc)[:300]}\nRepeat the evaluation and end with exactly the requested JSON object.")
        except Exception as exc:  # noqa: BLE001 - network, rate limits, empty replies
            # Invalid key, no permission or wrong endpoint: retrying would not help,
            # so abort the whole run with a clear message.
            if _is_config_error(exc):
                raise ConfigError(f"judge {job.judge.name!r}: {exc}") from exc
            last = exc
            if not is_last:
                time.sleep(min(60.0, 2.0 * 2 ** attempt) * random.uniform(0.5, 1.0))
        else:
            # The write is outside the try, so a disk error is raised instead of being
            # mistaken for a network error and causing new (paid) calls to the model.
            record.update(raw_text=resp.text, verdict=verdict.model_dump(), usage=resp.usage, error=None)
            _write(path, record)
            return "ok"

    # All attempts failed: save the error (and the last reply), so the call is visible
    # and retried on the next run.
    record.update(error=f"failed after {cfg.max_retries} attempts: {last}", raw_text=last_text)
    _write(path, record)
    return "error"


# =============================================================================
# 6. RUNNING ALL THE CALLS
# =============================================================================

def run_all(
        pairs,
        judges: list[Provider],
        prompt: Prompt,
        cfg: RunConfig,
        out_dir: str | Path,
        *,
        force: bool = False,
        limit: int | None = None
) -> dict[str, int]:
    """
    Run all the calls in parallel (``cfg.concurrency`` threads) and count the outcomes.
    :param pairs: the pairs to evaluate
    :param judges: the judges
    :param prompt: the evaluation prompt
    :param cfg: run options
    :param out_dir: results folder
    :param force: if True, ignore the cache and repeat all the calls
    :param limit: if given, run only the first ``limit`` pairs, each with every judge, order and
                  repetition (so no pair is left half evaluated)
    :return: ``{"ok": ..., "cached": ..., "error": ...}``
    :raises ConfigError: if two calls would share a file or a call hits a configuration problem;
                         the calls not started yet are cancelled
    """
    out_dir = Path(out_dir)
    # `limit` now applies to pairs, not to calls ([:None] keeps the whole list).
    pairs = list(pairs)[:limit]
    jobs = build_jobs(pairs, judges, prompt, cfg)

    # fail before any call if two jobs would overwrite each other's file.
    _check_paths(jobs, out_dir)
    counts = {"ok": 0, "cached": 0, "error": 0}

    # Threads suit this work: the time is spent waiting for the APIs, not computing.
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:

        # All jobs are submitted at once; the dict maps each future back to its job.
        futs = {pool.submit(run_job, j, cfg, out_dir, force): j for j in jobs}
        try:
            # as_completed yields the futures as they finish (not in submission order).
            for i, fut in enumerate(as_completed(futs), 1):
                status = fut.result()  # ConfigError propagates and aborts the run
                counts[status] += 1
                if status == "error":
                    log.warning("job failed: %s", futs[fut].path(out_dir).name)

                # Progress every 10 calls and at the end.
                if i % 10 == 0 or i == len(jobs):
                    log.info("progress %d/%d %s", i, len(jobs), counts)
        except BaseException:
            # on ConfigError (or Ctrl+C) cancel the calls still in the queue; without this,
            # leaving the `with` block would wait for all of them to run (and spend tokens).
            # The calls already running (at most `concurrency`) are allowed to finish.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return counts


# =============================================================================
# 7. LOADING THE SAVED CALLS
# =============================================================================

def load_records(
        out_dir: str | Path,
        *,
        judges: Iterable[str] | None = None,
        pair_ids: Iterable[str] | None = None,
        orders: Iterable[str] | None = None,
        reps: int | None = None
) -> list[dict]:
    """
    Read the successful calls saved in ``out_dir/runs``. Files that are unreadable or that
    contain an error (failed calls) are skipped. The optional filters keep only the calls of a
    given run, so files left over from earlier runs (removed judges or pairs, fewer orders or
    repetitions) are ignored.
    :param out_dir: results folder
    :param judges: if given, keep only these judge names
    :param pair_ids: if given, keep only these pairs
    :param orders: if given, keep only these presentation orders
    :param reps: if given, keep only repetitions ``0 .. reps-1``
    :return: one dict per call, with ``job``, ``cache_key``, ``raw_text``, ``verdict`` and ``usage``
    """
    # Filters, to drop calls that do not belong to the current run.
    judge_set = set(judges) if judges is not None else None
    pair_set = set(pair_ids) if pair_ids is not None else None
    order_set = set(orders) if orders is not None else None
    recs = []

    # runs/<judge>/<file>.json, sorted for a stable order (".tmp" files are not matched).
    for p in sorted((Path(out_dir) / "runs").glob("*/*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # FIX: also unreadable files and bad encodings
            log.warning("skipping unreadable file %s: %s", p, exc)
            continue

        # Keep only successful calls (no error and a verdict present).
        if not isinstance(r, dict) or r.get("error") or "verdict" not in r:
            continue
        job = r.get("job") or {}
        if judge_set is not None and job.get("judge") not in judge_set:
            continue
        if pair_set is not None and job.get("pair_id") not in pair_set:
            continue
        if order_set is not None and job.get("order") not in order_set:
            continue
        if reps is not None and not (isinstance(job.get("rep"), int) and 0 <= job["rep"] < reps):
            continue
        recs.append(r)
    return recs


def load_run_records(
        out_dir: str | Path
) -> list[dict]:
    """
    Read the successful calls that belong to the run described in ``out_dir/config.json``:
    its judges, pairs, orders and repetitions.
    :param out_dir: results folder created by ``Evaluator.evaluate``
    :return: one dict per call, as in :func:`load_records`
    :raises RuntimeError: if config.json is missing, unreadable or incomplete
    """
    out = Path(out_dir)
    try:
        meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
        opts = meta["options"]
        return load_records(out, judges=[j["name"] for j in meta["judges"]],
                            pair_ids=meta["pair_ids"], orders=opts["orders"], reps=opts["reps"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"{out} has no valid config.json (was it created by Evaluator.evaluate?): {exc}") from exc

def _is_config_error(exc: Exception) -> bool:
    """
    Tell whether an API error is a configuration problem that retrying cannot fix
    (invalid key, no permission, wrong endpoint/deployment/model).
    :param exc: the exception raised by the provider
    :return: True for HTTP 401, 403 and 404
    """
    # openai uses `status_code`, google-genai uses `code`.
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return code in (401, 403, 404)