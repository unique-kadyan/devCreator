"""LLM provider protocol and structured-output helpers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError as PydValidationError

from ..core.errors import ValidationError

M = TypeVar("M", bound=BaseModel)


@dataclass
class Completion:
    text: str
    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: dict = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, role: str = "story",
                 max_tokens: int = 4096, temperature: float = 0.9) -> Completion: ...


_FENCE_FULL = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_FENCE_OPEN = re.compile(r"^\s*```(?:json|JSON)?\s*", re.I)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")


def extract_json(text: str) -> str:
    """Pull JSON out of a reply that may be fenced, prefixed with prose, or truncated.

    Free models wrap their output constantly, and a reply cut off at max_tokens has an
    OPENING fence with no closing one - which is exactly the case a naive ```...``` regex
    misses, leaving the fence in the string and producing "expected value at line 1
    column 1". So: strip a fence from either end independently, then scan brackets.
    """
    m = _FENCE_FULL.search(text)
    if m:
        text = m.group(1)
    else:
        # Unbalanced fence: strip whichever side is present.
        text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text))
    text = text.strip()

    # Whichever bracket appears FIRST wins. Trying "{" first would truncate a top-level
    # array whose first element is an object - e.g. a scene list - to just that object.
    candidates = [(text.find(o), o, c) for o, c in (("{", "}"), ("[", "]"))
                  if text.find(o) != -1]
    for start, opener, closer in sorted(candidates):
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        # Never closed: the reply was cut off. Hand back everything from the opener so the
        # parse error names the real problem instead of complaining about leading prose.
        return text[start:]
    return text


def salvage_truncated_json(text: str) -> str | None:
    """Rescue a reply that was cut off at max_tokens.

    A scene list truncated part-way through its last element is not a failed generation -
    it is eight good scenes and one bad one. Discarding all nine costs a request against a
    50/day budget and usually comes back truncated again, because the model is verbose for
    the same reason it was verbose the first time.

    The rewind point is the innermost open ARRAY, not the innermost open container: an
    array holds interchangeable elements, so dropping an incomplete one leaves valid JSON,
    whereas truncating an object mid-way leaves it missing required keys. Returns None when
    there is nothing to salvage.
    """
    stack: list[str] = []          # expected closers, outermost first
    clean: list[int] = []          # per-depth offset where that container was last whole
    in_str = esc = False
    pairs = {"{": "}", "[": "]"}

    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in pairs:
            stack.append(pairs[c])
            clean.append(i + 1)                     # empty container is "whole"
        elif c in ("}", "]"):
            if stack and stack[-1] == c:
                stack.pop()
                clean.pop()
            if clean:
                clean[-1] = i + 1                   # the parent gained a complete element
        elif c == "," and clean:
            clean[-1] = i                           # cut here to drop what follows

    if not stack:
        return None                                 # nothing left open: not truncated
    depth = next((d for d in range(len(stack) - 1, -1, -1) if stack[d] == "]"), None)
    if depth is None:
        depth = 0                                   # no array at all: salvage the object
    cut = clean[depth]
    if cut <= 0:
        return None
    repaired = text[:cut].rstrip().rstrip(",")
    return repaired + "".join(reversed(stack[:depth + 1]))


def parse_model(text: str, model: type[M]) -> M:
    payload = extract_json(text)
    try:
        return model.model_validate_json(payload)
    except (PydValidationError, json.JSONDecodeError) as first:
        salvaged = salvage_truncated_json(payload)
        if salvaged is not None:
            try:
                return model.model_validate_json(salvaged)
            except (PydValidationError, json.JSONDecodeError):
                pass                    # salvage did not help; report the original error
        if isinstance(first, json.JSONDecodeError):
            raise ValidationError(f"{model.__name__}: not valid JSON - {first}") from first
        raise ValidationError(f"{model.__name__}: {first}") from first
