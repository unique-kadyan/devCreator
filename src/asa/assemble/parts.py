"""Splitting one story into release parts.

Parts are computed from *measured* scene durations, not from the model's `duration_hint_s`
and not by asking the model to write to a length. Hints are routinely out by a factor of
two - the authoritative duration only exists after the audio stage has rendered the voice
lines - and a free-tier model told to "write about three minutes" will not comply reliably
enough to build a release schedule on. Deterministic arithmetic over real durations fails
predictably; a model asked to count seconds fails silently.

Scenes are atomic. A part boundary can only fall *between* scenes, so a story whose scenes
are chunky may produce parts outside the requested window - that is a fact about the
material, not a failure, and `plan_parts` reports the real durations so the caller can say
so rather than pretend.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from ..core.logging import get_logger

log = get_logger("parts")


@dataclass(frozen=True)
class Part:
    index: int                 # 1-based, as viewers see it
    scenes: list[int]          # 0-based positions into the scene list, in order
    duration_s: float

    @property
    def label(self) -> str:
        return f"Part {self.index}"


def plan_parts(durations: Sequence[float], *, min_s: float = 150.0,
               max_s: float = 180.0) -> list[Part]:
    """Group scenes into parts of roughly `min_s`..`max_s` each.

    The count is chosen first and the boundaries are then balanced across it, rather than
    greedily filling parts to `max_s` and taking whatever is left. Greedy filling reliably
    produces a runt final part - three scenes and forty seconds - which is the worst thing
    to end a series on and cannot be fixed after the fact without re-cutting everything.
    """
    ds = [max(0.0, float(d)) for d in durations]
    if not ds:
        return []
    total = sum(ds)
    target = (min_s + max_s) / 2.0

    # How many parts the material can actually carry. Capping by `total // min_s` is what
    # stops a 4-minute story being split into two 2-minute halves that both undershoot.
    n = max(1, round(total / target)) if target > 0 else 1
    n = max(1, min(n, int(total // min_s) if min_s > 0 else n, len(ds)))

    if n == 1:
        return [Part(1, list(range(len(ds))), round(total, 2))]

    # Boundaries at the scene gap nearest each k*total/n, never reusing a gap and always
    # leaving at least one scene for every remaining part.
    cuts: list[int] = []
    cum: list[float] = []
    running = 0.0
    for d in ds:
        running += d
        cum.append(running)

    for k in range(1, n):
        want = total * k / n
        lo = (cuts[-1] + 1) if cuts else 1
        hi = len(ds) - (n - k)
        if lo > hi:
            break
        best = min(range(lo, hi + 1), key=lambda i: abs(cum[i - 1] - want))
        cuts.append(best)

    bounds = [0, *cuts, len(ds)]
    parts: list[Part] = []
    for i in range(len(bounds) - 1):
        idxs = list(range(bounds[i], bounds[i + 1]))
        if not idxs:
            continue
        parts.append(Part(len(parts) + 1, idxs,
                          round(sum(ds[j] for j in idxs), 2)))

    outside = [p.index for p in parts if not (min_s <= p.duration_s <= max_s)]
    if outside:
        # Not an error. Scene granularity decides what is reachable, and a caller that
        # wants to warn about it needs to know which parts and by how much.
        log.info("parts_outside_window", parts=outside, min_s=min_s, max_s=max_s,
                 durations=[p.duration_s for p in parts])
    return parts


def part_title(title: str, part: Part, total_parts: int, max_len: int = 100) -> str:
    """`Title (Part 2 of 3)`, trimming the title rather than the part marker.

    The marker is the part of the string doing real work - it is how a viewer knows there
    is more and which one they are on - so when the combination exceeds YouTube's 100
    character limit the story title gives way, never the marker.
    """
    if total_parts <= 1:
        return title[:max_len]
    marker = f" (Part {part.index} of {total_parts})"
    room = max_len - len(marker)
    return (title[:room].rstrip(" -–—,:;") + marker) if room > 0 else marker[:max_len]
