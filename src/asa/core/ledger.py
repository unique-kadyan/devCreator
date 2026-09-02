"""Asset provenance ledger.

The rule from docs/05 §5: no file enters a timeline without a licence row, and nothing
non-commercial is ever publishable. `audit()` is the query QC runs before every upload, and
it fails closed - an unknown licence blocks, it does not warn.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

COMMERCIAL_OK = {"CC0", "CC-BY", "YT-AUDIO-LIB", "PIXABAY", "APACHE-2.0", "OPENRAIL-PP-M"}
NEEDS_ATTRIBUTION = {"CC-BY"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def add_asset(db: Path, path: Path, kind: str, source: str, license_code: str,
              attribution: str | None = None, source_ref: str | None = None,
              meta: dict | None = None) -> int:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if license_code in NEEDS_ATTRIBUTION and not attribution:
        raise ValueError(
            f"{license_code} requires attribution; pass --attribution "
            f'"Title by Author (source) - CC BY 4.0". Omitting it voids the licence.')
    con = _connect(db)
    lic = con.execute("SELECT id FROM licenses WHERE license_code=?", (license_code,)).fetchone()
    if lic is None:
        raise ValueError(f"unknown licence code {license_code!r}; "
                         f"add it to the licenses table first")
    try:
        rel = str(path.relative_to(Path(db).resolve().parents[1]))
    except ValueError:
        rel = str(path)
    cur = con.execute(
        """INSERT INTO assets (kind, path, sha256, source, source_ref, license_id,
                               attribution, usage_allowed, meta, bytes)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(path) DO UPDATE SET
             kind=excluded.kind, sha256=excluded.sha256, source=excluded.source,
             source_ref=excluded.source_ref, license_id=excluded.license_id,
             attribution=excluded.attribution, meta=excluded.meta, bytes=excluded.bytes
           RETURNING id""",
        (kind, rel, _sha256(path), source, source_ref, lic["id"], attribution,
         1 if license_code in COMMERCIAL_OK else 0,
         json.dumps(meta or {}), path.stat().st_size))
    rid = cur.fetchone()[0]
    con.commit()
    con.close()
    return rid


def audit(db: Path, paths: list[str] | None = None) -> list[dict]:
    """Return every asset that must NOT be published. Empty list means clear."""
    con = _connect(db)
    q = ("""SELECT a.path, a.attribution, l.license_code, l.usage_allowed,
                   l.attribution_required
            FROM assets a LEFT JOIN licenses l ON l.id = a.license_id""")
    params: tuple = ()
    if paths:
        q += f" WHERE a.path IN ({','.join('?' * len(paths))})"
        params = tuple(paths)
    problems = []
    for r in con.execute(q, params):
        if r["license_code"] is None:
            problems.append({"path": r["path"], "reason": "no licence row"})
        elif r["usage_allowed"] != "commercial":
            problems.append({"path": r["path"],
                             "reason": f"licence {r['license_code']} is {r['usage_allowed']}"})
        elif r["attribution_required"] and not r["attribution"]:
            problems.append({"path": r["path"],
                             "reason": f"{r['license_code']} requires attribution, none stored"})
    con.close()
    return problems


def unregistered(db: Path, dirs: list[Path],
                 suffixes: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")) -> list[str]:
    """Files sitting on disk that have never been registered. These are the real risk."""
    con = _connect(db)
    known = {r[0] for r in con.execute("SELECT path FROM assets")}
    con.close()
    root = Path(db).resolve().parents[1]
    out = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue
            try:
                rel = str(p.resolve().relative_to(root))
            except ValueError:
                rel = str(p)
            if rel not in known:
                out.append(rel)
    return out


def attribution_block(db: Path, paths: list[str]) -> str:
    """The credits text that must appear in the YouTube description."""
    if not paths:
        return ""
    con = _connect(db)
    rows = con.execute(
        f"""SELECT a.attribution FROM assets a
            JOIN licenses l ON l.id=a.license_id
            WHERE l.attribution_required=1 AND a.path IN ({','.join('?' * len(paths))})
            AND a.attribution IS NOT NULL""", tuple(paths)).fetchall()
    con.close()
    lines = sorted({r["attribution"] for r in rows})
    return ("Credits:\n" + "\n".join(lines)) if lines else ""
