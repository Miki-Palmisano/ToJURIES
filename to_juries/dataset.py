"""The report pairs to evaluate."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pair:
    pair_id: str
    task_prompt: str
    reference_text: str
    cutoff: str
    baseline_text: str  # first system (e.g. PRE)
    candidate_text: str  # second system (e.g. POST)


def _read(base: Path, rel: str) -> str:
    p = Path(rel)
    p = p if p.is_absolute() else base / p
    if not p.exists():
        raise FileNotFoundError(f"file referenced in the CSV not found: {p}")
    return p.read_text(encoding="utf-8")


class PairDataset(Sequence):
    """A list of :class:`Pair`. Build it from a CSV with :meth:`from_csv`.

    CSV columns: pair_id, task_prompt (or task_file), reference_file, cutoff,
    baseline_file, candidate_file. Relative paths are resolved against the CSV's folder.
    """

    def __init__(self, pairs: list[Pair]) -> None:
        if not pairs:
            raise ValueError("the dataset is empty")
        if len({p.pair_id for p in pairs}) != len(pairs):
            raise ValueError("pair_id values must be unique")
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]

    @classmethod
    def from_csv(cls, path: str | Path) -> "PairDataset":
        path = Path(path)
        base = path.parent
        pairs: list[Pair] = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            missing = {"pair_id", "reference_file", "cutoff", "baseline_file", "candidate_file"} - cols
            if missing:
                raise ValueError(f"the CSV is missing columns: {sorted(missing)}")
            if not ({"task_prompt", "task_file"} & cols):
                raise ValueError("the CSV needs a 'task_prompt' or 'task_file' column")
            for i, row in enumerate(reader, start=2):
                pid = (row.get("pair_id") or "").strip()
                if not pid:
                    raise ValueError(f"CSV line {i}: empty pair_id")
                task = (row.get("task_prompt") or "").strip()
                if not task and row.get("task_file"):
                    task = _read(base, row["task_file"]).strip()
                if not task:
                    raise ValueError(f"CSV line {i}: empty task")
                pairs.append(Pair(
                    pair_id=pid, task_prompt=task,
                    reference_text=_read(base, row["reference_file"]),
                    cutoff=(row.get("cutoff") or "").strip(),
                    baseline_text=_read(base, row["baseline_file"]),
                    candidate_text=_read(base, row["candidate_file"]),
                ))
        return cls(pairs)
