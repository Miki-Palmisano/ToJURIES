"""The evaluation prompt. Only the five placeholders are filled; the text is never altered."""

# Enables modern type annotations (e.g. `str | Path`) on older Python versions.
from __future__ import annotations

import hashlib
import re
from importlib import resources
from pathlib import Path


# =============================================================================
# 1. PLACEHOLDERS
# =============================================================================

# The five placeholders the prompt must contain, written as {NAME} in the text.
PLACEHOLDERS = ("TASK_PROMPT", "ARTICLE_TEXT", "REPORT_A", "REPORT_B", "TASK_CUTOFF_DATE")

# Regex matching exactly one of the five placeholders, e.g. "{REPORT_A}";
# the name is captured in group 1. Other braces in the text (e.g. the JSON
# example in the prompt) do not match and are left as they are.
_PATTERN = re.compile(r"\{(" + "|".join(PLACEHOLDERS) + r")\}")

# The package holding the default prompt is derived from this module's own package,
# so renaming the package does not break Prompt.default(). The fallback covers the case
# where the module is loaded outside its package.
_PROMPTS_PACKAGE = f"{__package__ or 'to_juries'}.prompts"


# =============================================================================
# 2. THE PROMPT
# =============================================================================

class Prompt:
    """``Prompt(text)``, ``Prompt.from_file(path)`` or ``Prompt.default()``.

    The text must contain {TASK_PROMPT}, {ARTICLE_TEXT}, {REPORT_A}, {REPORT_B}, {TASK_CUTOFF_DATE}
    and must ask for the final JSON object (report_A / report_B ...) that the library reads.
    """

    def __init__(
            self,
            text: str
    ) -> None:
        """
        Create a prompt from its text, checking that all five placeholders are present.
        :param text: full text of the prompt
        :raises TypeError: if ``text`` is not a string
        :raises ValueError: if one or more placeholders are missing
        """
        # Explicit type check, so e.g. Prompt(None) gives a clear error
        # instead of a TypeError raised from inside the placeholder check.
        if not isinstance(text, str):
            raise TypeError(f"the prompt text must be a string, got {type(text).__name__}")

        # A missing placeholder would mean the judge never sees that piece of input
        # (e.g. one of the two reports): fail early, listing all the missing ones.
        missing = [p for p in PLACEHOLDERS if "{" + p + "}" not in text]
        if missing:
            raise ValueError(f"the prompt is missing placeholders: {', '.join('{'+m+'}' for m in missing)}")
        self.text = text

    # --- Alternative constructors --------------------------------------------

    @classmethod
    def from_file(
            cls,
            path: str | Path
    ) -> "Prompt":
        """
        Create a prompt from a text file (UTF-8). Line endings are normalised to "\\n"
        (Windows "\\r\\n" included), so the same prompt has the same hash on every system.
        :param path: path of the file containing the prompt
        :return: the prompt
        :raises FileNotFoundError: if the file does not exist
        :raises ValueError: if the text misses one or more placeholders
        """
        # read_text converts "\r\n" to "\n"; this is now stated in the
        # docstring, since it means the hash differs from a hash of the raw file bytes.
        return cls(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def default(cls) -> "Prompt":
        """
        Create the prompt shipped with the library (``prompts/evaluation.md``), unchanged.
        :return: the default prompt
        """
        # importlib.resources reads the file from inside the installed package,
        # so it works wherever the package is installed (also from a wheel/zip).
        # FIX: package name no longer hard-coded.
        return cls(resources.files(_PROMPTS_PACKAGE).joinpath("evaluation.md").read_text(encoding="utf-8"))

    # --- Filling the placeholders --------------------------------------------

    def render(
            self,
            *,
            task_prompt: str,
            reference: str,
            cutoff: str,
            report_a: str,
            report_b: str
    ) -> str:
        """
        Fill the five placeholders of the prompt for one call, leaving the rest of the text unchanged.
        The substitution is done in a single pass: if a report contains something like "{REPORT_B}",
        it is not replaced again.
        :param task_prompt: research task, replaces {TASK_PROMPT}
        :param reference: expert reference article, replaces {ARTICLE_TEXT}
        :param cutoff: evaluation cutoff date, replaces {TASK_CUTOFF_DATE}
        :param report_a: text shown as REPORT A, replaces {REPORT_A}
        :param report_b: text shown as REPORT B, replaces {REPORT_B}
        :return: the prompt ready to be sent to the judge
        """
        # (The `*` makes all arguments keyword-only, so REPORT A and B cannot be swapped by position.)
        # Placeholder name -> value for this call.
        values = {"TASK_PROMPT": task_prompt, "ARTICLE_TEXT": reference, "REPORT_A": report_a,
                  "REPORT_B": report_b, "TASK_CUTOFF_DATE": cutoff}

        # single pass: text that itself contains "{REPORT_B}" is never re-substituted
        # (unlike chained str.replace calls). Using a function as replacement also means
        # backslashes in the texts are inserted literally, not read as regex escapes.
        return _PATTERN.sub(lambda m: values[m.group(1)], self.text)

    # --- Fingerprint ---------------------------------------------------------

    @property
    def hash(self) -> str:
        """
        Short fingerprint of the prompt text. It is saved in ``config.json`` and is part of the
        cache key, so changing the prompt makes the calls run again instead of reusing old results.
        :return: the first 16 characters of the SHA-256 of the text
        """
        # 16 hex characters = 64 bits: more than enough to tell prompt versions apart.
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]