"""Character ids are canonical, ASCII and unambiguous.

Two functions computed this id: story/generator.py used str.isalnum(), which treats
Devanagari as alphanumeric, and characters/factory.py stripped to [a-z0-9]. With an English
cast they agreed and nothing showed. With a Hindi cast the story referenced
'श_य_म_cow' while the factory built 'char_cow', and saving the story failed on a foreign
key after the whole story had been generated.
"""
import pytest

from asa.characters.factory import slug
from asa.story.generator import _slug


NAMES = ["Milo", "Old Man Mercer", "Ravi the Bold", "श्यामा", "मोती", "चीकू", "",
         "  spaced  out  ", "O'Brien", "Zoë"]


@pytest.mark.parametrize("name", NAMES)
def test_both_call_sites_agree(name):
    assert _slug(name, "cow") == slug(name, "cow")


@pytest.mark.parametrize("name", NAMES)
def test_ids_are_ascii(name):
    # The id becomes a directory name, a database key and part of an image prompt.
    ident = slug(name, "cow")
    assert ident.isascii(), f"{name!r} produced a non-ASCII id {ident!r}"
    assert " " not in ident and "/" not in ident


def test_distinct_hindi_names_do_not_collide():
    # The original fallback made every Hindi name 'char_<species>', silently merging two
    # characters into one row, one puppet and one voice.
    ids = {slug(n, "cow") for n in ("श्यामा", "मोती", "चीकू", "गौरी")}
    assert len(ids) == 4, f"names collapsed together: {ids}"


def test_the_same_name_is_always_the_same_id():
    # Character reuse across episodes depends on this being stable.
    assert slug("श्यामा", "cow") == slug("श्यामा", "cow")
    assert slug("Milo", "fox") == slug("Milo", "fox")


def test_species_separates_identical_names():
    assert slug("मोती", "cow") != slug("मोती", "dog")


def test_empty_name_still_produces_a_usable_id():
    ident = slug("", "cow")
    assert ident.endswith("_cow") and len(ident) > 4
