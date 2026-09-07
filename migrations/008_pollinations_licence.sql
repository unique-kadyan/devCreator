-- Pollinations output, admitted to publication by an explicit operator decision.
--
-- Not folded into UNKNOWN and not relabelled CC0. UNKNOWN must stay fail-closed, because
-- its job is to stop the NEXT provider nobody has thought about yet; and calling this CC0
-- would put a false statement in the provenance record, which is the one thing a ledger
-- exists not to do. So it gets its own code, and the notes say exactly what was decided.
--
-- The decision (2026-09-04, repo operator): Pollinations requires no key and publishes no
-- clear grant of commercial reuse. The operator reviewed this and chose to publish on it
-- anyway, accepting the risk, because the alternative was an unwatchable episode - the
-- HuggingFace free image allowance is exhausted after ~4 stills and everything else falls
-- to the procedural placeholder.
--
-- To reverse it: set usage_allowed back to 'unknown' here and flip
-- providers.image.pollinations.enabled to false. Assets already registered keep this code,
-- so `asa assets` still shows precisely which files were published under it.
INSERT OR IGNORE INTO licenses
    (license_code, attribution_required, attribution_text, usage_allowed, license_url, notes)
VALUES
    ('POLLINATIONS-TOS-ACCEPTED', 0, NULL, 'commercial',
     'https://pollinations.ai/',
     'Terms do not clearly grant commercial reuse. Operator accepted the risk 2026-09-04. Reversible: see migrations/008.');
