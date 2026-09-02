#!/usr/bin/env bash
# Back up the database and the irreplaceable assets.
#
# `sqlite3 .backup` rather than `cp`: the runner may be mid-write, and copying a WAL
# database's main file alone produces a backup that restores to a state that never existed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ASA_BACKUP_DIR:-$ROOT/data/backups}"
KEEP="${ASA_BACKUP_KEEP:-14}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$DEST"

sqlite3 "$ROOT/data/asa.db" ".backup '$DEST/asa-$STAMP.db'"
gzip -9 -f "$DEST/asa-$STAMP.db"

# Characters and backgrounds are the only assets that cannot be regenerated identically:
# a puppet rebuilt from a changed palette is a different character, and a re-rolled plate
# breaks continuity with every episode that already used it.
tar -czf "$DEST/assets-$STAMP.tar.gz" \
    -C "$ROOT" assets/characters assets/backgrounds 2>/dev/null || true

# config/.env and the YouTube token are NOT backed up here on purpose. Copying secrets
# into a directory with laxer permissions is how they leak. Back them up yourself, to
# somewhere encrypted.

find "$DEST" -name 'asa-*.db.gz' -type f -printf '%T@ %p\n' | sort -rn | tail -n +"$((KEEP+1))" \
  | cut -d' ' -f2- | xargs -r rm -f
find "$DEST" -name 'assets-*.tar.gz' -type f -printf '%T@ %p\n' | sort -rn | tail -n +"$((KEEP+1))" \
  | cut -d' ' -f2- | xargs -r rm -f

echo "backup complete: $DEST/asa-$STAMP.db.gz"
