"""Series metadata: how a part presents itself on YouTube."""
from asa.assemble.parts import Part, part_title
from asa.publish.metadata import MAX_DESCRIPTION, part_description


class TestPartDescription:
    def test_single_part_is_untouched(self):
        assert part_description("Story body.", 1, 1) == "Story body."

    def test_position_is_stated_first(self):
        out = part_description("Story body.", 2, 3)
        # YouTube collapses a description after ~3 lines; a pointer below the fold is
        # invisible to the viewer who most needs it.
        assert out.splitlines()[0] == "Part 2 of 3."

    def test_links_back_to_earlier_parts(self):
        out = part_description("Story body.", 3, 3, ["aaa111", "bbb222"])
        assert "https://youtu.be/aaa111" in out
        assert "https://youtu.be/bbb222" in out

    def test_first_part_invites_the_subscribe_instead_of_linking_back(self):
        out = part_description("Story body.", 1, 3, [])
        assert "subscribe" in out.lower()
        assert "youtu.be" not in out

    def test_final_part_with_no_known_ids_does_not_promise_more(self):
        out = part_description("Story body.", 3, 3, [])
        assert "next part" not in out.lower()

    def test_empty_ids_are_dropped(self):
        out = part_description("Story body.", 3, 3, [None, "", "ccc333"])
        assert "https://youtu.be/ccc333" in out
        assert "youtu.be/ " not in out

    def test_stays_within_the_description_limit(self):
        out = part_description("x" * (MAX_DESCRIPTION + 500), 2, 3, ["aaa111"])
        assert len(out) <= MAX_DESCRIPTION


class TestPartTitleInSeries:
    def test_parts_are_numbered_for_the_viewer(self):
        titles = [part_title("The Lantern Promise", Part(i, [], 160.0), 3)
                  for i in (1, 2, 3)]
        assert titles == ["The Lantern Promise (Part 1 of 3)",
                          "The Lantern Promise (Part 2 of 3)",
                          "The Lantern Promise (Part 3 of 3)"]

    def test_titles_are_distinct(self):
        # Identical titles across a series read as duplicate uploads in a subscriber feed.
        titles = {part_title("A Story", Part(i, [], 160.0), 4) for i in range(1, 5)}
        assert len(titles) == 4
