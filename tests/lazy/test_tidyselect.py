"""
uv run pytest tests/lazy/test_tidyselect.py
"""

import tidypyrs as tp


def test_contains_ignore_case():
    """Can find columns that contain and ignores case"""
    tl = tp.TibbleLazy({"name": ["a", "a", "b"], "NUMBER": [2, 1, 1]})
    actual = tl.select(tp.contains("M", True))
    print(actual)
    expected = tp.TibbleLazy({"name": ["a", "a", "b"], "NUMBER": [2, 1, 1]})
    assert actual.equals(expected), "contains ignore case failed"


def test_contains_include_case():
    """Can find columns that contain and includes case"""
    tl = tp.TibbleLazy({"name": ["a", "a", "b"], "NUMBER": [2, 1, 1]})
    actual = tl.select(tp.contains("M", ignore_case=False))
    expected = tp.TibbleLazy({"NUMBER": [2, 1, 1]})
    assert actual.equals(expected), "contains includes case failed"


def test_contains_treats_match_as_literal():
    tl = tp.TibbleLazy({"a.b": [1], "axb": [2]})
    actual = tl.select(tp.contains("."))
    expected = tp.TibbleLazy({"a.b": [1]})
    assert actual.equals(expected)


def test_ends_with_ignore_case():
    """Can find columns that ends_with and ignores case"""
    tl = tp.TibbleLazy({"writer": ["a", "a", "b"], "NUMBER": [2, 1, 1]})
    actual = tl.select(tp.ends_with("er", True))
    expected = tp.TibbleLazy({"writer": ["a", "a", "b"], "NUMBER": [2, 1, 1]})
    assert actual.equals(expected), "ends_with ignore case failed"


def test_ends_with_include_case():
    """Can find columns that ends_with and ignores case"""
    tl = tp.TibbleLazy({"writer": ["a", "a", "b"], "NUMBER": [2, 1, 1]})
    actual = tl.select(tp.ends_with("er", ignore_case=False))
    expected = tp.TibbleLazy({"writer": ["a", "a", "b"]})
    assert actual.equals(expected), "ends_with ignore case failed"


def test_everything():
    """Can find all columns"""
    tl = tp.TibbleLazy({"name": ["a", "a", "b"], "value": [2, 1, 1]})
    actual = tl.select(tp.everything())
    expected = tp.TibbleLazy({"name": ["a", "a", "b"], "value": [2, 1, 1]})
    assert actual.equals(expected), "everything failed"


def test_starts_with_ignore_case():
    """Can find columns that starts_with and ignores case"""
    tl = tp.TibbleLazy({"name": ["a", "a", "b"], "Number": [2, 1, 1]})
    actual = tl.select(tp.starts_with("n", True))
    expected = tp.TibbleLazy({"name": ["a", "a", "b"], "Number": [2, 1, 1]})
    assert actual.equals(expected), "starts_with ignore case failed"


def test_starts_with_include_case():
    """Can find columns that starts_with and includes case"""
    tl = tp.TibbleLazy({"name": ["a", "a", "b"], "Number": [2, 1, 1]})
    actual = tl.select(tp.starts_with("n", ignore_case=False))
    expected = tp.TibbleLazy({"name": ["a", "a", "b"]})
    assert actual.equals(expected), "starts_with include case failed"


def test_where():
    """Can use where"""
    tl = tp.TibbleLazy({"string_col": ["a"], "numeric_col": [1]})
    actual = tl.select(tp.where("string"))
    expected = tl.select("string_col")
    assert actual.equals(expected), "where('numeric') failed"
    actual = tl.select(tp.where("numeric"))
    expected = tl.select("numeric_col")
    assert actual.equals(expected), "where('numeric') failed"
