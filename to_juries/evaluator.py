"""Evaluator: define judges, prompt and options once, then evaluate a dataset."""

# Enables modern type annotations (e.g. `str | Path`, `dict[str, int]`) on older Python versions.
from __future__ import annotations

import json
from pathlib import Path

from .analysis.pipeline import analyze as _analyze
from .analysis.tables import system_of
from .config import ConfigError, RunConfig
from .dataset import PairDataset
from .prompt import Prompt
from .judje import Provider
from .runner import load_run_records, run_all


# =============================================================================
# 1. RESULT OF AN EVALUATION
# =============================================================================

class Run:
    """Result of an evaluation: a folder on disk."""

    def __init__(
            self,
            out_dir: Path,
            counts: dict[str, int]
    ) -> None:
        """
        Handle on the result of an evaluation (the results live on disk, in ``out_dir``).
        :param out_dir: folder where the evaluation wrote its files
        :param counts: number of calls ``{"ok", "cached", "error"}``: new successful calls,
                       calls reused from the cache and calls that failed
        """
        # The object holds no results in memory: only the folder and the call counts.
        # Everything else is read from disk when needed.
        self.out_dir = out_dir
        self.counts = counts  # {"ok", "cached", "error"}

    def records(self) -> list[dict]:
        """
        Load the successful calls of this run (judges, pairs, orders and repetitions in config.json).
        :return: one dict per call, with its ``job`` (pair, judge, order, rep) and the parsed ``verdict``
        """
        # FIX: leftover files from earlier runs in the same folder are ignored.
        return load_run_records(self.out_dir)

    def analyze(self) -> dict:
        """
        Analyse the results saved in the folder (local, no calls to the models, no tokens used).
        Writes report.md and the CSV tables into the folder.
        :return: the main tables: ``scores``, ``summary``, ``gain_stats``, ``agreement``,
                 ``flags`` and ``votes`` (pandas DataFrames)
        :raises RuntimeError: if the folder has no successful calls
        """
        # Options and judges are not passed: the analysis reads them from config.json in the folder.
        return _analyze(self.out_dir)

    def __repr__(self) -> str:
        """
        Short description of the run, shown when the object is printed.
        :return: a string with the folder and the call counts
        """
        # !r adds quotes around the path, e.g. Run(out_dir='results', counts={...}).
        return f"Run(out_dir={str(self.out_dir)!r}, counts={self.counts})"


# =============================================================================
# 2. THE EVALUATOR
# =============================================================================

class Evaluator:
    """
    ``Evaluator(judges=[...], prompt=Prompt.default(), labels=("PRE", "POST"))``

    ``evaluate(dataset, out_dir)`` calls every judge on every pair in every presentation order
    and writes into ``out_dir``:

      config.json   judges (no keys), options, prompt hash
      prompt.md     the prompt used
      runs/<judge>/<pair>__<order>__r<rep>.json   raw reply + parsed JSON, one per call
      results.json  all parsed verdicts, with the system behind REPORT A / REPORT B
    """

    def __init__(
            self,
            judges: list[Provider],
            prompt: Prompt | None = None,
            **options
    ) -> None:
        """
        Define the judges, the prompt and the options once; then use ``evaluate`` on any dataset.
        :param judges: the models that evaluate (e.g. ``AzureOpenAI`` and ``Gemini``), with unique names
        :param prompt: the evaluation prompt; if None, the default prompt is used
        :param options: run options of :class:`RunConfig`: ``labels``, ``orders``, ``reps``,
                        ``concurrency``, ``max_retries``, ``bootstrap``, ``alpha``, ``seed``
        :raises ConfigError: if there are no judges, the judge names repeat or an option is invalid
        """
        # --- Judges ----------------------------------------------------------
        # At least one judge is needed, otherwise there is nothing to run.
        if not judges:
            raise ConfigError("at least one judge is required")
        # Names must be unique: they are used as folder names (runs/<judge>/)
        # and as keys in the analysis.
        names = [j.name for j in judges]
        if len(set(names)) != len(names):
            raise ConfigError(f"judge names must be unique, got {names}")
        # Copy into a new list, so later changes to the caller's list do not affect the evaluator.
        self.judges = list(judges)

        # --- Prompt and options ----------------------------------------------
        self.prompt = prompt or Prompt.default()
        # Any validation error from pydantic is re-raised as ConfigError,
        # so callers only need to catch one exception type for configuration problems.
        # `from exc` keeps the original error in the traceback.
        try:
            self.config = RunConfig(**options)
        except Exception as exc:
            raise ConfigError(f"invalid options: {exc}") from exc

    def evaluate(
            self,
            dataset: PairDataset,
            out_dir: str | Path = "results",
            *,
            force: bool = False,
            limit: int | None = None,
            analyze: bool = False
    ) -> Run:
        """
        Call every judge on every pair, in every presentation order and repetition (uses tokens).
        Finished calls are cached in ``out_dir`` and not repeated; failed ones are retried.
        :param dataset: the pairs to evaluate
        :param out_dir: folder where config, prompt, one JSON per call and results.json are written
        :param force: if True, ignore the cache and repeat all the calls
        :param limit: if given, run only the first ``limit`` pairs (useful for a quick test);
                      results.json and the analysis then cover only those pairs
        :param analyze: if True, run the analysis at the end
        :return: the :class:`Run` of this folder, with the call counts
        :raises ConfigError: if ``out_dir`` already holds calls made with a different prompt or
                             labels (and ``force`` is False), if two calls would share a file, or for
                             a configuration problem that makes the calls fail (e.g. a missing API key)
        """
        # (The `*` makes force, limit and analyze keyword-only arguments.)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Refuse to mix cached calls made with another prompt or other labels
        # (must run before config.json is overwritten below).
        if not force:
            self._check_existing(out)

        # `limit` is applied here, to pairs; the ids of the pairs in this run are
        # saved in config.json, so results.json and the analysis read only their calls.
        pairs = list(dataset)[:limit]

        # --- Save what defines the run, before any call ----------------------
        # The prompt text, to know exactly what the judges were asked.
        (out / "prompt.md").write_text(self.prompt.text, encoding="utf-8")
        # Options, judge descriptions (without API keys), prompt hash and pairs of the run:
        # this is what the analysis reads later, and what makes the run reproducible.
        (out / "config.json").write_text(json.dumps({
            "options": self.config.model_dump(),
            "judges": [j.describe() for j in self.judges],
            "prompt_hash": self.prompt.hash,
            "pair_ids": [p.pair_id for p in pairs],
        }, indent=1), encoding="utf-8")

        # --- Run the calls ---------------------------------------------------
        # run_all handles cache, concurrency and retries, and returns the counts {"ok", "cached", "error"}.
        counts = run_all(pairs, self.judges, self.prompt, self.config, out, force=force)

        # --- Collect and (optionally) analyse --------------------------------
        # Only the calls of this run (judges, pairs, orders and repetitions in config.json).
        records = load_run_records(out)
        self._write_results(out, records)
        run = Run(out, counts)
        # The analysis runs only if requested and if at least one call succeeded
        # (otherwise analyze() would raise RuntimeError).
        if analyze and records:
            run.analyze()
        return run

    def _check_existing(
            self,
            out: Path
    ) -> None:
        """
        Check that the calls already saved in ``out`` were made with the same prompt and labels,
        so the cache does not mix verdicts from different setups.
        :param out: the results folder
        :raises ConfigError: if the folder holds calls and its config.json has a different
                             prompt hash or different labels
        """
        cfg_path = out / "config.json"
        # Nothing to compare: new folder, or a folder with no saved calls.
        if not cfg_path.exists() or not (out / "runs").exists():
            return
        try:
            old = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return  # unreadable config: it will be overwritten
        changed = []
        if old.get("prompt_hash") != self.prompt.hash:
            changed.append("prompt")
        # JSON stores the tuple as a list: compare as tuples.
        old_labels = tuple(old.get("options", {}).get("labels", ()))
        if old_labels and old_labels != tuple(self.config.labels):
            changed.append(f"labels {old_labels} -> {tuple(self.config.labels)}")
        if changed:
            raise ConfigError(
                f"{out} already holds calls made with a different {', '.join(changed)}; "
                "use a new out_dir or pass force=True")

    def _write_results(
            self,
            out: Path,
            records: list[dict]
    ) -> None:
        """
        Write ``results.json``: all the parsed verdicts, each with the label (e.g. PRE/POST)
        that was shown as REPORT A and as REPORT B in that call.
        :param out: the results folder
        :param records: the successful calls, as returned by ``load_records``
        """
        rows = []
        for rec in records:
            job = rec["job"]
            # Job metadata + which label was behind REPORT A and REPORT B in this call
            # (it depends on the order) + the judge's parsed verdict.
            rows.append({**job,
                         "report_A": system_of("A", job["order"], self.config.labels),
                         "report_B": system_of("B", job["order"], self.config.labels),
                         "verdict": rec["verdict"]})
        # ensure_ascii=False keeps accented and non-Latin characters readable in the file.
        (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")