"""The report pairs to evaluate."""

# Enables modern type annotations (e.g. `str | Path`, `list[Pair]`) on older Python versions.
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 1. ONE PAIR TO EVALUATE
# =============================================================================

# frozen=True: the pair is immutable (fields cannot be changed after creation)
# and hashable, so it cannot be modified by mistake during the run.
@dataclass(frozen=True)
class Pair:
    pair_id: str  # unique identifier of the pair
    task_prompt: str  # the task both systems had to carry out
    reference_text: str  # reference material the judges compare the reports against
    cutoff: str  # temporal cutoff (used to check temporal leakage)
    first_text: str  # first system (e.g. PRE)
    second_text: str  # second system (e.g. POST)

# =============================================================================
# 2. READING TEXT FILES
# =============================================================================

def _read(
        base: Path,
        rel: str
) -> str:
    """
    Read a text file referenced in a pair.
    :param base: folder used to resolve relative paths (the JSON file's folder or ``base_dir``)
    :param rel: path of the file, absolute or relative to ``base``
    :return: the content of the file (UTF-8)
    :raises FileNotFoundError: if the file does not exist
    """
    p = Path(rel)

    # Absolute paths are used as they are; relative ones are joined to `base`.
    p = p if p.is_absolute() else base / p

    # Explicit check, so the error message says the file was referenced in the pairs.
    if not p.exists():
        raise FileNotFoundError(f"file referenced in the pairs not found: {p}")
    return p.read_text(encoding="utf-8")

# =============================================================================
# 3. BUILDING A PAIR FROM A RECORD
# =============================================================================

# For each text field of Pair: the key holding the text inline in the record
# and the alternative key holding the path of a file with that text.
_TEXT_FIELDS = {  # Pair field -> (inline key, file key)
    "task_prompt": ("task_prompt", "task_file"),
    "reference_text": ("reference_text", "reference_file"),
    "first_text": ("first_text", "first_file"),
    "second_text": ("second_text", "second_file"),
}


def _pair_from_row(
        row: dict,
        base: Path,
        where: str
) -> Pair:
    """
    Build one :class:`Pair` from a dict. Each text can be given inline (e.g. ``first_text``)
    or as a file path (e.g. ``first_file``); the inline text wins if both are present.
    The same applies to ``task_prompt``/``task_file`` and ``reference_*``.
    :param row: record with ``pair_id``, ``cutoff`` and the texts (or their file paths)
    :param base: folder used to resolve relative file paths
    :param where: label of the record (e.g. "record 3"), used in error messages
    :return: the pair, with all the texts loaded
    :raises ValueError: if ``pair_id`` is empty or a text is missing (neither inline nor file)
    :raises FileNotFoundError: if a referenced file does not exist
    """
    # pair_id is required: None, missing or blank -> error.
    # str() also accepts numeric ids (e.g. 3 -> "3").
    pid = str(row.get("pair_id") or "").strip()
    if not pid:
        raise ValueError(f"{where}: empty pair_id")
    texts = {}

    for field, (inline, file_key) in _TEXT_FIELDS.items():
        # 1) Inline text, if present (None is treated as empty).
        value = (row.get(inline) or "").strip() if inline in row else ""

        # 2) Otherwise, read the file given in the file key (if any).
        if not value and row.get(file_key):
            value = _read(base, row[file_key])

        # 3) Neither source gave a non-blank text -> error naming both keys.
        if not value.strip():
            raise ValueError(f"{where} (pair {pid}): missing {inline} or {file_key}")

        # The task prompt is always stripped; the other texts keep the content as read
        # (inline texts were already stripped in step 1, texts from files are not).
        texts[field] = value.strip() if field == "task_prompt" else value

    # cutoff is required: without it the judge cannot assess temporal leakage.
    cutoff = str(row.get("cutoff") or "").strip()
    if not cutoff:
        raise ValueError(f"{where} (pair {pid}): empty cutoff")
    return Pair(pair_id=pid, cutoff=cutoff, **texts)

# =============================================================================
# 4. THE DATASET
# =============================================================================

# Subclassing Sequence and implementing __len__ and __getitem__ is enough
# to get iteration, `in`, index(), count() and reversed() for free.
class PairDataset(Sequence):
    """A list of :class:`Pair`.

    Build it with :meth:`from_records` (list of dicts defined in your script) or
    :meth:`from_json` (a JSON file with the same records). Per pair: ``pair_id``, ``cutoff`` and,
    for each of task / reference / frist (first system) / second (second system), either
    the text (``task_prompt``, ``reference_text``, ``first_text``, ``second_text``) or a
    file path (``task_file``, ``reference_file``, ``first_file``, ``second_file``).
    Relative paths are resolved against the JSON file's folder (or ``base_dir``).
    """

    def __init__(
            self,
            pairs: list[Pair]
    ) -> None:
        """
        Create the dataset from already built pairs (usually via ``from_records`` or ``from_json``).
        :param pairs: the pairs to evaluate
        :raises ValueError: if the list is empty or ``pair_id`` values are not unique
        """
        # An empty dataset would produce a run with no calls.
        if not pairs:
            raise ValueError("the dataset is empty")

        # pair_id must be unique: the analysis groups and pivots by pair_id.
        if len({p.pair_id for p in pairs}) != len(pairs):
            raise ValueError("pair_id values must be unique")

        # Copy into a new list, so later changes to the caller's list do not affect the dataset.
        self.pairs = list(pairs)

    def __len__(self) -> int:
        """
        Number of pairs in the dataset.
        :return: the number of pairs
        """
        return len(self.pairs)

    def __getitem__(self, i):
        """
        Access to the pairs by position, so the dataset can be indexed, sliced and iterated.
        :param i: index or slice
        :return: a :class:`Pair`, or a list of pairs if ``i`` is a slice
        """
        # Delegates to the internal list (a slice returns a plain list, not a PairDataset).
        return self.pairs[i]

    # --- Alternative constructors --------------------------------------------

    @classmethod
    def from_records(
            cls,
            records: list[dict],
            base_dir: str | Path = "."
    ) -> "PairDataset":
        """
        Build the dataset from a list of dicts defined in your script.
        :param records: one dict per pair (see :class:`PairDataset` for the keys)
        :param base_dir: folder used to resolve relative file paths
        :return: the dataset
        :raises ValueError: if a record is invalid, the list is empty or ``pair_id`` values repeat
        :raises FileNotFoundError: if a referenced file does not exist
        """
        base = Path(base_dir)

        # Records are numbered from 1 in error messages ("record 1", "record 2", ...).
        return cls([_pair_from_row(r, base, f"record {i}") for i, r in enumerate(records, start=1)])

    @classmethod
    def from_json(
            cls,
            path: str | Path
    ) -> "PairDataset":
        """
        Build the dataset from a JSON file holding a list of records (or ``{"pairs": [...]}``).
        Relative file paths inside the records are resolved against the JSON file's folder.
        :param path: path of the JSON file
        :return: the dataset
        :raises ValueError: if a record is invalid, the list is empty or ``pair_id`` values repeat
        :raises FileNotFoundError: if the JSON file or a referenced file does not exist
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        # Two accepted formats: a plain list of records, or an object with a "pairs" key.
        records = data["pairs"] if isinstance(data, dict) else data

        # Relative paths are resolved against the JSON file's folder,
        # so the dataset works wherever the script is run from.
        return cls.from_records(records, base_dir=path.parent)