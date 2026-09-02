"""Hashtag merge rules.

The two numbers under test are YouTube's, not ours: only the first three hashtags render
above the title, and a sixteenth hashtag makes YouTube ignore all of them. Both fail
silently in production, which is exactly why they are pinned here.
"""
from asa.publish.metadata import (
    DISPLAY_HASHTAGS, MAX_DESCRIPTION, MAX_HASHTAGS, build_description, clean_hashtags,
    normalise_hashtag,
)

STORY = {"title": "t", "logline": "l", "hook": "h", "setting": "s", "moral": "m"}
CAST = [{"name": "Milo", "species": "fox"}]


class TestNormalise:
    def test_strips_hash_and_keeps_word(self):
        assert normalise_hashtag("#fox") == "fox"

    def test_multiword_becomes_camelcase_not_truncated(self):
        # "#river rescue" would tag only "river" and leave "rescue" as loose text.
        assert normalise_hashtag("river rescue") == "RiverRescue"

    def test_punctuation_is_a_word_break(self):
        assert normalise_hashtag("lantern-bridge!") == "LanternBridge"

    def test_rejects_too_short_and_pure_digits(self):
        assert normalise_hashtag("ab") == ""
        assert normalise_hashtag("2026") == ""
        assert normalise_hashtag("###") == ""
        assert normalise_hashtag("   ") == ""

    def test_caps_length(self):
        assert len(normalise_hashtag("a" * 80)) == 30


class TestMerge:
    def test_group_order_is_preserved(self):
        out = clean_hashtags(["Animation"], ["fox"], ["ShortFilm"])
        assert out == ["Animation", "fox", "ShortFilm"]

    def test_lead_group_owns_the_three_displayed_slots(self):
        lead = ["Animation", "AnimalStories", "MiloAndFriends"]
        out = clean_hashtags(lead, ["fox", "owl"], [])
        assert out[:DISPLAY_HASHTAGS] == lead

    def test_dedupes_case_insensitively(self):
        # "#Animation" and "#animation" are one tag to YouTube; spending two of fifteen
        # slots on it is the bug this guards.
        out = clean_hashtags(["Animation"], ["animation", "ANIMATION"], [])
        assert out == ["Animation"]

    def test_hard_cap_at_fifteen(self):
        out = clean_hashtags([f"tag{i:02d}" for i in range(40)])
        assert len(out) == MAX_HASHTAGS == 15

    def test_cap_applies_across_groups(self):
        out = clean_hashtags(["lead"] * 1, [f"story{i:02d}" for i in range(30)],
                             ["evergreen"])
        assert len(out) == MAX_HASHTAGS
        assert "evergreen" not in out          # the cap bites before the last group

    def test_tolerates_empty_and_none_groups(self):
        assert clean_hashtags([], None, ["fox"]) == ["fox"]


class TestDescription:
    def _build(self, hashtags, attribution="", cta=""):
        return build_description(STORY, "Base copy.", CAST, "Milo & Friends",
                                 attribution, False, hashtags, cta)

    def test_hashtags_are_rendered_with_a_single_hash(self):
        out = self._build(["fox", "owl"])
        assert out.rstrip().endswith("#fox #owl")

    def test_all_hashtags_are_emitted_not_just_three(self):
        tags = [f"tag{i:02d}" for i in range(10)]
        out = self._build(tags)
        for t in tags:
            assert f"#{t}" in out

    def test_cta_appears_above_the_signoff(self):
        out = self._build(["fox"], cta="Subscribe for more.")
        assert out.index("Subscribe for more.") < out.index("— Milo & Friends")

    def test_no_cta_leaves_no_blank_gap(self):
        assert "\n\n\n" not in self._build(["fox"], cta="   ")

    def test_hashtags_survive_a_description_that_needs_truncating(self):
        # A long licence block used to push the hashtag line past MAX_DESCRIPTION, costing
        # the video its discovery surface with nothing in the logs to say so.
        out = self._build(["fox", "owl"], attribution="x" * 9000)
        assert len(out) <= MAX_DESCRIPTION
        assert out.rstrip().endswith("#fox #owl")

    def test_empty_hashtags_still_produces_a_description(self):
        out = self._build([])
        assert "Base copy." in out and "#" not in out
