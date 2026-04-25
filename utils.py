"""
utils.py — Shared Utility Functions
====================================
Houses helpers for JSON parsing, logging, formatting, and
other cross-cutting concerns so domain modules stay clean.
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone

# ────────────────────────────────────────────────────────────
# Logger Setup
# ────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create and return a consistently-formatted logger.

    Parameters
    ----------
    name  : module name (typically __name__)
    level : one of DEBUG, INFO, WARNING, ERROR, CRITICAL

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ────────────────────────────────────────────────────────────
# Safe JSON Parsing
# ────────────────────────────────────────────────────────────

def safe_parse_json(raw: str) -> dict | None:
    """
    Attempt to extract and parse JSON from a potentially noisy
    LLM response.  Handles:
      • Raw valid JSON
      • JSON wrapped in markdown fences (```json ... ```)
      • JSON embedded in surrounding prose

    Returns
    -------
    dict if parsing succeeds, else None.
    """
    # 1. Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Try to strip markdown code fences
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(fence_pattern, raw)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Try to find the first { ... } block
    brace_pattern = r"\{[\s\S]*\}"
    match = re.search(brace_pattern, raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ────────────────────────────────────────────────────────────
# Formatting Helpers
# ────────────────────────────────────────────────────────────

def timestamp_now() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def truncate(text: str, max_len: int = 280) -> str:
    """Truncate *text* to *max_len* characters, appending '…' if trimmed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def print_separator(title: str = "", width: int = 60) -> None:
    """Print a visual separator for console output."""
    if title:
        pad = max(0, width - len(title) - 4)
        left = pad // 2
        right = pad - left
        print(f"\n{'=' * left} {title} {'=' * right}")
    else:
        print(f"\n{'=' * width}")


def format_similarity_table(results: list[dict]) -> str:
    """
    Pretty-print routing results as a markdown-style table.

    Parameters
    ----------
    results : list of dicts with keys 'persona_id', 'name', 'similarity'

    Returns
    -------
    str : formatted table
    """
    lines = [
        "| Rank | Bot                | Similarity |",
        "|------|--------------------|------------|",
    ]
    for idx, r in enumerate(results, start=1):
        name = r["name"].ljust(18)
        sim = f'{r["similarity"]:.4f}'.rjust(10)
        lines.append(f"| {idx:<4} | {name} | {sim} |")
    return "\n".join(lines)
