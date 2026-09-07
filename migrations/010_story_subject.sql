-- Real subject matter: what an episode claims about the world, and when and where it is set.
--
-- Three columns rather than one JSON blob, because they are read by three different things
-- at three different times:
--
--   subject  a human reviewing the approval queue needs to see at a glance that this
--            episode asserts something, before watching seven minutes to find out;
--   facts    the claims themselves, as a JSON array. This is the audit trail. Nothing in
--            this pipeline can verify that a claim is TRUE - no LLM check would be worth
--            trusting for that - so what the column buys is that a reviewer checks a list
--            against a source instead of transcribing the finished audio;
--   period   the era AND place, and it is here because it is consumed by the ART stage,
--            which runs in a different process (and after a crash, on a different day) from
--            the story stage that decided it. `channel.region_hint` pins every generation
--            to contemporary India, which is right for this channel and wrong for Syracuse
--            in 212 BC; a non-empty period replaces that hint.
--
-- All three default to empty, which is what every existing row means: fiction, present day.

ALTER TABLE stories ADD COLUMN subject TEXT NOT NULL DEFAULT '';
ALTER TABLE stories ADD COLUMN facts   TEXT NOT NULL DEFAULT '[]';
ALTER TABLE stories ADD COLUMN period  TEXT NOT NULL DEFAULT '';
