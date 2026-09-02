"""
uv run pytest tests/lazy/test_stringr.py
"""

import tidypyrs as tp
from tidypyrs import col as c


def test_str_paste():
    """Can use paste"""
    tl = tp.TibbleLazy(x=["a", "b", "c"])
    actual = tl.mutate(x_end=tp.str_paste(c("x"), "end", sep="_"))
    expected = tp.TibbleLazy(x=["a", "b", "c"], x_end=["a_end", "b_end", "c_end"])
    assert actual.equals(expected), "paste failed"


def test_str_paste0():
    """Can use paste0"""
    tl = tp.TibbleLazy(x=["a", "b", "c"])
    actual = tl.mutate(x_end=tp.str_paste0(c("x"), "_end"))
    expected = tp.TibbleLazy(x=["a", "b", "c"], x_end=["a_end", "b_end", "c_end"])
    assert actual.equals(expected), "paste0 failed"


def test_str_concat():
    """Can use str_c"""
    tl = tp.TibbleLazy(x=["a", "b", "c"])
    actual = tl.mutate(x_end=tp.str_concat(c("x"), "end", sep="_"))
    expected = tp.TibbleLazy(x=["a", "b", "c"], x_end=["a_end", "b_end", "c_end"])
    assert actual.equals(expected), "str_c failed"


def test_str_detect_single():
    """Can str_detect find a single string"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(
        x=tp.str_detect("name", "a"), y=tp.str_detect("name", "a", negate=True)
    )
    expected = tp.TibbleLazy(
        name=["apple", "banana", "pear", "grape"],
        x=[True, True, True, True],
        y=[False, False, False, False],
    )
    assert actual.equals(expected), "str_detect single failed"


def test_str_detect_multiple():
    """Can str_detect find multiple strings"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(
        x=tp.str_detect("name", ["a", "e"]),
        y=tp.str_detect("name", ["a", "e"], negate=True),
    )
    expected = tp.TibbleLazy(
        name=["apple", "banana", "pear", "grape"],
        x=[True, False, True, True],
        y=[False, True, False, False],
    )
    assert actual.equals(expected), "str_detect multiple failed"


def test_str_ends():
    """Can use str_end"""
    tl = tp.TibbleLazy(words=["apple", "bear", "amazing"])
    actual = tl.filter(tp.str_ends(c("words"), "ing"))
    expected = tp.TibbleLazy(words=["amazing"])
    assert actual.equals(expected), "str_ends failed"


def test_str_extract():
    """Can str_extract extract strings"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(x=tp.str_extract("name", "pp"))
    expected = tp.TibbleLazy(
        name=["apple", "banana", "pear", "grape"], x=["pp", None, None, None]
    )
    assert actual.equals(expected), "str_extract failed"


def test_str_length():
    """Can str_length count strings"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(x=tp.str_length("name"))
    expected = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"], x=[5, 6, 4, 5])
    assert actual.equals(expected), "str_length failed"


def test_str_sub():
    """Can str_sub can extract strings"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(x=tp.str_sub("name", 0, 3))
    expected = tp.TibbleLazy(
        name=["apple", "banana", "pear", "grape"], x=["app", "ban", "pea", "gra"]
    )
    assert actual.equals(expected), "str_sub failed"


def test_str_remove_all():
    """Can str_remove_all find all strings and remove"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(new_name=tp.str_remove_all(c("name"), "a"))
    expected = tp.TibbleLazy(
        name=["apple", "banana", "pear", "grape"],
        new_name=["pple", "bnn", "per", "grpe"],
    )
    assert actual.equals(expected), "str_remove_all failed"


def test_str_remove():
    """Can str_remove finds first instance of string and remove"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(new_name=tp.str_remove(c("name"), "a"))
    expected = tp.TibbleLazy(
        name=["apple", "banana", "pear", "grape"],
        new_name=["pple", "bnana", "per", "grpe"],
    )
    assert actual.equals(expected), "str_remove failed"


def test_str_replace_all():
    """Can str_replace_all find all strings and replace"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(new_name=tp.str_replace_all(c("name"), "a", "A"))
    expected = tp.TibbleLazy(
        name=["apple", "banana", "pear", "grape"],
        new_name=["Apple", "bAnAnA", "peAr", "grApe"],
    )
    assert actual.equals(expected), "str_replace_all failed"


def test_str_replace():
    """Can str_replace finds first instance of string and replace"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(new_name=tp.str_replace(c("name"), "a", "A"))
    expected = tp.TibbleLazy(
        name=["apple", "banana", "pear", "grape"],
        new_name=["Apple", "bAnana", "peAr", "grApe"],
    )
    assert actual.equals(expected), "str_replace failed"


def test_str_starts():
    """Can use str_starts"""
    tl = tp.TibbleLazy(words=["apple", "bear", "amazing"])
    actual = tl.filter(tp.str_starts(c("words"), "a"))
    expected = tp.TibbleLazy(words=["apple", "amazing"])
    assert actual.equals(expected), "str_starts failed"


def test_str_to_lower():
    """Can str_to_lower lowercase a string"""
    tl = tp.TibbleLazy(name=["APPLE", "BANANA", "PEAR", "GRAPE"])
    actual = tl.mutate(name=tp.str_to_lower(c("name")))
    expected = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    assert actual.equals(expected), "str_to_lower failed"


def test_str_to_upper():
    """Can str_to_upper uppercase a string"""
    tl = tp.TibbleLazy(name=["apple", "banana", "pear", "grape"])
    actual = tl.mutate(name=tp.str_to_upper(c("name")))
    expected = tp.TibbleLazy(name=["APPLE", "BANANA", "PEAR", "GRAPE"])
    assert actual.equals(expected), "str_to_upper failed"


def test_str_trim():
    """Can str_to_upper uppercase a string"""
    tl = tp.TibbleLazy(x=[" a ", " b ", " c "])
    actual = tl.mutate(
        both=tp.str_trim("x"),
        left=tp.str_trim("x", "left"),
        right=tp.str_trim("x", "right"),
    ).drop("x")
    expected = tp.TibbleLazy(
        both=["a", "b", "c"], left=["a ", "b ", "c "], right=[" a", " b", " c"]
    )
    assert actual.equals(expected), "str_trim failed"
