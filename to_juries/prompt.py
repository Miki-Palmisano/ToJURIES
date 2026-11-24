"""The evaluation prompt. Only the five placeholders are filled; the text is never altered."""

from __future__ import annotations

import hashlib
import re
from importlib import resources
from pathlib import Path

PLACEHOLDERS = ("TASK_PROMPT", "ARTICLE_TEXT", "REPORT_A", "REPORT_B", "TASK_CUTOFF_DATE")
_PATTERN = re.compile(r"\{(" + "|".join(PLACEHOLDERS) + r")\}")


class Prompt:
    """``Prompt(text)``, ``Prompt.from_file(path)`` or ``Prompt.default()``.

    The text must contain {TASK_PROMPT}, {ARTICLE_TEXT}, {REPORT_A}, {REPORT_B}, {TASK_CUTOFF_DATE}
    and must ask for the final JSON object (report_A / report_B ...) that the library reads.
    """

    def __init__(self, text: str) -> None:
        missing = [p for p in PLACEHOLDERS if "{" + p + "}" not in text]
        if missing:
            raise ValueError(f"the prompt is missing placeholders: {', '.join('{'+m+'}' for m in missing)}")
        self.text = text

    @classmethod
    def from_file(cls, path: str | Path) -> "Prompt":
        return cls(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def default(cls) -> "Prompt":
        return cls(resources.files("to_juries.prompts").joinpath("evaluation.md").read_text(encoding="utf-8"))

    def render(self, *, task_prompt: str, reference: str, cutoff: str, report_a: str, report_b: str) -> str:
        values = {"TASK_PROMPT": task_prompt, "ARTICLE_TEXT": reference, "REPORT_A": report_a,
                  "REPORT_B": report_b, "TASK_CUTOFF_DATE": cutoff}
        # single pass: text that itself contains "{REPORT_B}" is never re-substituted
        return _PATTERN.sub(lambda m: values[m.group(1)], self.text)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
