-- Licence codes for checkpoints run locally by media/images/local.py.
--
-- The local runner is one provider that can load several models, and their licences are
-- genuinely different - so the code is declared per image by the provider rather than read
-- from a table keyed on the provider name (see `GeneratedImage.license_code`). That is only
-- honest if every code it can declare exists here, because `ledger.add_asset` refuses a
-- code it does not know and would otherwise fail at publication time rather than at the
-- point the model was chosen.
--
-- Apache-2.0 already exists and covers SSD-1B, FLUX.1-schnell and Z-Image-Turbo, which are
-- the three models media/images/local.MODELS admits by default. Only Stable Diffusion 1.5
-- needs a new row.
--
-- CreativeML OpenRAIL-M is NOT the same licence as OpenRAIL++-M (already present as
-- OPENRAIL-PP-M, which covers the SD 2.x / SDXL family), so it gets its own code rather
-- than being folded into it. Both permit commercial use; they differ in their use-based
-- restrictions, and a provenance record that names the wrong one is a false statement about
-- what was actually granted.
--
-- Commercial use IS permitted by this licence. What it attaches are use-based restrictions
-- (Attachment A) on what may be generated - the operator agrees to them by publishing on
-- it, and this channel's content policy (docs/05-COMPLIANCE.md) is already narrower.
INSERT OR IGNORE INTO licenses
    (license_code, attribution_required, attribution_text, usage_allowed, license_url, notes)
VALUES
    ('CREATIVEML-OPENRAIL-M', 0, NULL, 'commercial',
     'https://huggingface.co/spaces/CompVis/stable-diffusion-license',
     'Stable Diffusion 1.5 weights. Commercial use permitted subject to the use-based restrictions in Attachment A. Distinct from OPENRAIL-PP-M.');
