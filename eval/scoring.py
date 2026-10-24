"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

ToolAlpaca scoring adapted from the SDFT authors' evaluator.

Returns 1 when the predicted action-name multiset and merged action inputs match
the gold actions and inputs; otherwise returns 0.

Main changes:
- Parses nested Action Input JSON objects with JSONDecoder.raw_decode.
- Normalizes language names and codes in language-valued fields.
- Ignores whitespace around commas in selected symbol/currency list fields.

These normalization rules extend the original evaluator's input matching.

Original implementation:
https://github.com/idanshen/
Self-Distillation/blob/main/eval_tooluse.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


_ACTION_INPUT_MARKER = re.compile(r"Action Input:\s*")

# Canonicalize only values attached to language-like keys.  The map intentionally uses
# common ISO-639-1 codes so both a full language name and its API-style short code compare
# identically without changing unrelated free-text fields.
_LANGUAGE_KEYS = {"language", "lang", "language_code", "languagecode"}
_LANGUAGE_ALIASES = {
    "arabic": "ar",
    "ar": "ar",
    "chinese": "zh",
    "zh": "zh",
    "czech": "cs",
    "cs": "cs",
    "danish": "da",
    "da": "da",
    "dutch": "nl",
    "nl": "nl",
    "english": "en",
    "en": "en",
    "finnish": "fi",
    "fi": "fi",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "greek": "el",
    "el": "el",
    "hebrew": "he",
    "he": "he",
    "hindi": "hi",
    "hi": "hi",
    "hungarian": "hu",
    "hu": "hu",
    "indonesian": "id",
    "id": "id",
    "italian": "it",
    "it": "it",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
    "norwegian": "no",
    "no": "no",
    "polish": "pl",
    "pl": "pl",
    "portuguese": "pt",
    "pt": "pt",
    "romanian": "ro",
    "ro": "ro",
    "russian": "ru",
    "ru": "ru",
    "spanish": "es",
    "es": "es",
    "swedish": "sv",
    "sv": "sv",
    "thai": "th",
    "th": "th",
    "turkish": "tr",
    "tr": "tr",
    "ukrainian": "uk",
    "uk": "uk",
    "vietnamese": "vi",
    "vi": "vi",
}

# These API fields conventionally encode a list as a comma-separated string.  Restricting
# whitespace normalization to such fields avoids treating punctuation differences in free
# text (comments, search text, etc.) as equivalent.
_COMMA_LIST_KEYS = {
    "symbols",
    "symbol_codes",
    "currency_codes",
    "currencies",
    "currency_pairs",
}


def extract_actions(text: str) -> list[str]:
    return re.findall(r"Action:\s*(\w+)", text)


def _decode_action_input(text: str, start: int) -> dict[str, Any] | None:
    """Decode one complete JSON object beginning at/after ``start``.

    ``JSONDecoder.raw_decode`` understands nested objects/arrays and quoted braces, unlike
    the previous ``{.*?}`` regex.  Malformed blocks are ignored, matching the evaluator's
    previous fail-soft behavior.
    """
    remainder = text[start:].lstrip()
    if not remainder:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(remainder)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_action_inputs(text: str) -> dict[str, Any]:
    combined_dict: dict[str, Any] = {}
    for match in _ACTION_INPUT_MARKER.finditer(text):
        parsed = _decode_action_input(text, match.end())
        if parsed is not None:
            combined_dict.update(parsed)
    return combined_dict


def _normalize_string_for_key(key: str | None, value: str) -> str:
    key_lower = (key or "").lower()

    if key_lower in _LANGUAGE_KEYS:
        stripped = value.strip()
        return _LANGUAGE_ALIASES.get(stripped.lower(), stripped)

    if key_lower in _COMMA_LIST_KEYS and "," in value:
        return ",".join(part.strip() for part in value.split(","))

    return value


def normalize_action_input(value: Any, *, key: str | None = None) -> Any:
    """Return a comparison-only canonical form for an Action Input value."""
    if isinstance(value, dict):
        return {k: normalize_action_input(v, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_action_input(item, key=key) for item in value]
    if isinstance(value, str):
        return _normalize_string_for_key(key, value)
    return value


def score_tooluse_response(response: str, golden_answer: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """Return ``(1 or 0, details)`` for one response against its gold actions."""
    pred_actions = extract_actions(response)
    pred_inputs = extract_action_inputs(response)

    gt_actions = [item["Action"] for item in golden_answer]
    gt_inputs: dict[str, Any] = {}
    for item in golden_answer:
        try:
            parsed = json.loads(item["Action_Input"])
            if isinstance(parsed, dict):
                gt_inputs.update(parsed)
        except Exception:
            pass

    pred_inputs_normalized = normalize_action_input(pred_inputs)
    gt_inputs_normalized = normalize_action_input(gt_inputs)

    actions_match = Counter(pred_actions) == Counter(gt_actions)
    inputs_match = pred_inputs_normalized == gt_inputs_normalized
    correct = int(actions_match and inputs_match)

    details = {
        "pred_actions": pred_actions,
        "pred_inputs": pred_inputs,
        "gt_actions": gt_actions,
        "gt_inputs": gt_inputs,
        "pred_inputs_normalized": pred_inputs_normalized,
        "gt_inputs_normalized": gt_inputs_normalized,
        "actions_match": actions_match,
        "inputs_match": inputs_match,
        "correct": bool(correct),
    }
    return correct, details
