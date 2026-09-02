"""Freesound sound-effect resolution.

Local library first; Freesound only on a miss. Results are filtered to CC0 and CC-BY at the
query level, every download is registered in the licence ledger, and anything that comes back
with an unexpected licence is discarded rather than used.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from ...core.errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from ...core.ledger import add_asset

API = "https://freesound.org/apiv2"
# Freesound licence strings -> our ledger codes. Anything not in this map is rejected.
ALLOWED = {
    "Creative Commons 0": "CC0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0",
    "Attribution": "CC-BY",
    "Attribution 4.0": "CC-BY",
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY",
}
LICENSE_FILTER = 'license:("Creative Commons 0" OR "Attribution" OR "Attribution 4.0")'


class SFXLibrary:
    """Resolve a scene's snake_case sfx tags to files on disk."""

    def __init__(self, db: Path, local_dir: Path, api_key: str | None = None,
                 timeout: float = 30.0):
        self.db = db
        self.local_dir = local_dir
        self.api_key = api_key or ""
        self.timeout = timeout

    def tags(self) -> list[str]:
        """Tags already in the local library. Fed to the story prompt so the model reuses
        effects we own instead of inventing ones we would have to fetch."""
        if not self.local_dir.exists():
            return []
        return sorted({p.stem.split("__")[0] for p in self.local_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in
                       (".wav", ".mp3", ".ogg", ".flac", ".m4a")})

    def local(self, tag: str) -> Path | None:
        for suffix in (".wav", ".mp3", ".flac", ".ogg"):
            p = self.local_dir / f"{tag}{suffix}"
            if p.exists():
                return p
        return None

    def resolve(self, tag: str) -> Path | None:
        """Local hit, else Freesound, else None. Never raises for a missing sound -
        a scene without a footstep still ships."""
        hit = self.local(tag)
        if hit:
            return hit
        if not self.api_key:
            return None
        try:
            return self.fetch(tag)
        except (ProviderError, AuthError):
            return None

    def fetch(self, tag: str) -> Path | None:
        query = tag.replace("_", " ")
        headers = {"Authorization": f"Token {self.api_key}"}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            r = client.get(f"{API}/search/text/", params={
                "query": query, "filter": f"{LICENSE_FILTER} duration:[0.2 TO 12]",
                "fields": "id,name,license,previews,username,url,duration",
                "sort": "score", "page_size": 5})
            self._raise_for_status(r)
            results = r.json().get("results") or []
            for item in results:
                code = ALLOWED.get((item.get("license") or "").strip())
                if code is None:
                    continue          # unexpected licence - skip, never "probably fine"
                url = (item.get("previews") or {}).get("preview-hq-mp3")
                if not url:
                    continue
                self.local_dir.mkdir(parents=True, exist_ok=True)
                dest = self.local_dir / f"{tag}.mp3"
                d = client.get(url)
                self._raise_for_status(d)
                dest.write_bytes(d.content)
                attribution = (f'"{item["name"]}" by {item["username"]} - '
                               f'freesound.org - {"CC0" if code == "CC0" else "CC BY 4.0"}')
                add_asset(self.db, dest, kind="sfx", source="freesound",
                          license_code=code,
                          attribution=attribution if code == "CC-BY" else None,
                          source_ref=item.get("url"),
                          meta={"freesound_id": item["id"], "duration": item.get("duration")})
                return dest
        return None

    @staticmethod
    def _raise_for_status(r: httpx.Response) -> None:
        if r.status_code == 200:
            return
        if r.status_code in (401, 403):
            raise AuthError(f"freesound rejected the API key ({r.status_code})")
        if r.status_code == 429:
            raise RateLimited("freesound rate limit (60/min, 2000/day)",
                              provider="freesound", retry_after_s=20.0)
        if r.status_code == 409:
            raise QuotaExhausted("freesound daily quota exhausted", provider="freesound")
        raise ProviderError(f"freesound HTTP {r.status_code}: {r.text[:120]}",
                            provider="freesound")
