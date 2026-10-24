"""
uv run pytest tests/lazy/test_join.py
"""

import pytest

import tidypyrs as tp


def test_join_automatic_keys():
    """Can automatically join using common columns"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tl1.join(tl2).arrange("id")

    expected = tp.TibbleLazy(
        {
            "id": [2, 3],
            "x": ["b", "c"],
            "y": ["B", "C"],
        }
    )

    assert actual.equals(expected), "automatic inner join failed"


def test_join_inner():
    """Can perform an inner join"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tl1.join(
        tl2,
        on="id",
        how="inner",
    ).arrange("id")

    expected = tp.TibbleLazy(
        {
            "id": [2, 3],
            "x": ["b", "c"],
            "y": ["B", "C"],
        }
    )

    assert actual.equals(expected), "inner join failed"


def test_join_left():
    """Can perform a left join"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tl1.join(
        tl2,
        on="id",
        how="left",
    ).arrange("id")

    expected = tp.TibbleLazy(
        {
            "id": [1, 2, 3],
            "x": ["a", "b", "c"],
            "y": [None, "B", "C"],
        }
    )

    assert actual.equals(expected), "left join failed"


def test_join_right():
    """Can perform a right join"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tl1.join(
        tl2,
        on="id",
        how="right",
    ).arrange("id")

    expected = tp.TibbleLazy(
        {
            "x": ["b", "c", None],
            "id": [2, 3, 4],
            "y": ["B", "C", "D"],
        }
    )

    assert actual.equals(expected), "right join failed"


def test_join_full():
    """Can perform a full join"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tl1.join(
        tl2,
        on="id",
        how="full",
        coalesce=True,
    ).arrange("id")

    expected = tp.TibbleLazy(
        {
            "id": [1, 2, 3, 4],
            "x": ["a", "b", "c", None],
            "y": [None, "B", "C", "D"],
        }
    )

    assert actual.equals(expected), "full join failed"


def test_join_semi():
    """Can perform a semi join"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tl1.join(
        tl2,
        on="id",
        how="semi",
    ).arrange("id")

    expected = tp.TibbleLazy(
        {
            "id": [2, 3],
            "x": ["b", "c"],
        }
    )

    assert actual.equals(expected), "semi join failed"


def test_join_anti():
    """Can perform an anti join"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tl1.join(
        tl2,
        on="id",
        how="anti",
    ).arrange("id")

    expected = tp.TibbleLazy(
        {
            "id": [1],
            "x": ["a"],
        }
    )

    assert actual.equals(expected), "anti join failed"


def test_join_cross():
    """Can perform a cross join"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2], "x": ["a", "b"]}
    )
    tl2 = tp.TibbleLazy(
        {"group": ["A", "B"]}
    )

    actual = tl1.join(
        tl2,
        how="cross",
    ).arrange("id", "group")

    expected = tp.TibbleLazy(
        {
            "id": [1, 1, 2, 2],
            "x": ["a", "a", "b", "b"],
            "group": ["A", "B", "A", "B"],
        }
    )

    assert actual.equals(expected), "cross join failed"


def test_join_different_keys():
    """Can join using different key names"""
    tl1 = tp.TibbleLazy(
        {"left_id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tl2 = tp.TibbleLazy(
        {"right_id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tl1.join(
        tl2,
        left_on="left_id",
        right_on="right_id",
        how="inner",
    ).arrange("left_id")

    expected = tp.TibbleLazy(
        {
            "left_id": [2, 3],
            "x": ["b", "c"],
            "y": ["B", "C"],
        }
    )

    assert actual.equals(expected), "different-key join failed"


def test_join_suffix():
    """Can apply a suffix to duplicate columns"""
    tl1 = tp.TibbleLazy(
        {"id": [1, 2], "value": ["a", "b"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [1, 2], "value": ["A", "B"]}
    )

    actual = tl1.join(
        tl2,
        on="id",
        suffix="_other",
    ).arrange("id")

    expected = tp.TibbleLazy(
        {
            "id": [1, 2],
            "value": ["a", "b"],
            "value_other": ["A", "B"],
        }
    )

    assert actual.equals(expected), "join suffix failed"


def test_join_nulls_equal():
    """Can match null join keys"""
    tl1 = tp.TibbleLazy(
        {"id": [None, 1], "x": ["null-left", "one-left"]}
    )
    tl2 = tp.TibbleLazy(
        {"id": [None, 1], "y": ["null-right", "one-right"]}
    )

    actual = tl1.join(
        tl2,
        on="id",
        nulls_equal=True,
    ).arrange("id")

    expected = tp.TibbleLazy(
        {
            "id": [None, 1],
            "x": ["null-left", "one-left"],
            "y": ["null-right", "one-right"],
        }
    )

    assert actual.equals(expected), "null-key join failed"


def test_join_without_common_columns():
    """Raises when no join keys or common columns are available"""
    tl1 = tp.TibbleLazy({"x": [1, 2]})
    tl2 = tp.TibbleLazy({"y": [1, 2]})

    with pytest.raises(ValueError, match="No common columns"):
        tl1.join(tl2)
