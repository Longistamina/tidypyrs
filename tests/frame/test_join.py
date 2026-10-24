"""
uv run pytest tests/frame/test_join.py
"""

import pytest

import tidypyrs as tp


def test_join_automatic_keys():
    """Can automatically join using common columns"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tf1.join(tf2).arrange("id")

    expected = tp.TibbleFrame(
        {
            "id": [2, 3],
            "x": ["b", "c"],
            "y": ["B", "C"],
        }
    )

    assert actual.equals(expected), "automatic inner join failed"


def test_join_inner():
    """Can perform an inner join"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tf1.join(
        tf2,
        on="id",
        how="inner",
    ).arrange("id")

    expected = tp.TibbleFrame(
        {
            "id": [2, 3],
            "x": ["b", "c"],
            "y": ["B", "C"],
        }
    )

    assert actual.equals(expected), "inner join failed"


def test_join_left():
    """Can perform a left join"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tf1.join(
        tf2,
        on="id",
        how="left",
    ).arrange("id")

    expected = tp.TibbleFrame(
        {
            "id": [1, 2, 3],
            "x": ["a", "b", "c"],
            "y": [None, "B", "C"],
        }
    )

    assert actual.equals(expected), "left join failed"


def test_join_right():
    """Can perform a right join"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tf1.join(
        tf2,
        on="id",
        how="right",
    ).arrange("id")

    expected = tp.TibbleFrame(
        {
            "x": ["b", "c", None],
            "id": [2, 3, 4],
            "y": ["B", "C", "D"],
        }
    )

    assert actual.equals(expected), "right join failed"


def test_join_full():
    """Can perform a full join"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tf1.join(
        tf2,
        on="id",
        how="full",
        coalesce=True,
    ).arrange("id")

    expected = tp.TibbleFrame(
        {
            "id": [1, 2, 3, 4],
            "x": ["a", "b", "c", None],
            "y": [None, "B", "C", "D"],
        }
    )

    assert actual.equals(expected), "full join failed"


def test_join_semi():
    """Can perform a semi join"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tf1.join(
        tf2,
        on="id",
        how="semi",
    ).arrange("id")

    expected = tp.TibbleFrame(
        {
            "id": [2, 3],
            "x": ["b", "c"],
        }
    )

    assert actual.equals(expected), "semi join failed"


def test_join_anti():
    """Can perform an anti join"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tf1.join(
        tf2,
        on="id",
        how="anti",
    ).arrange("id")

    expected = tp.TibbleFrame(
        {
            "id": [1],
            "x": ["a"],
        }
    )

    assert actual.equals(expected), "anti join failed"


def test_join_cross():
    """Can perform a cross join"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2], "x": ["a", "b"]}
    )
    tf2 = tp.TibbleFrame(
        {"group": ["A", "B"]}
    )

    actual = tf1.join(
        tf2,
        how="cross",
    ).arrange("id", "group")

    expected = tp.TibbleFrame(
        {
            "id": [1, 1, 2, 2],
            "x": ["a", "a", "b", "b"],
            "group": ["A", "B", "A", "B"],
        }
    )

    assert actual.equals(expected), "cross join failed"


def test_join_different_keys():
    """Can join using different key names"""
    tf1 = tp.TibbleFrame(
        {"left_id": [1, 2, 3], "x": ["a", "b", "c"]}
    )
    tf2 = tp.TibbleFrame(
        {"right_id": [2, 3, 4], "y": ["B", "C", "D"]}
    )

    actual = tf1.join(
        tf2,
        left_on="left_id",
        right_on="right_id",
        how="inner",
    ).arrange("left_id")

    expected = tp.TibbleFrame(
        {
            "left_id": [2, 3],
            "x": ["b", "c"],
            "y": ["B", "C"],
        }
    )

    assert actual.equals(expected), "different-key join failed"


def test_join_suffix():
    """Can apply a suffix to duplicate columns"""
    tf1 = tp.TibbleFrame(
        {"id": [1, 2], "value": ["a", "b"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [1, 2], "value": ["A", "B"]}
    )

    actual = tf1.join(
        tf2,
        on="id",
        suffix="_other",
    ).arrange("id")

    expected = tp.TibbleFrame(
        {
            "id": [1, 2],
            "value": ["a", "b"],
            "value_other": ["A", "B"],
        }
    )

    assert actual.equals(expected), "join suffix failed"


def test_join_nulls_equal():
    """Can match null join keys"""
    tf1 = tp.TibbleFrame(
        {"id": [None, 1], "x": ["null-left", "one-left"]}
    )
    tf2 = tp.TibbleFrame(
        {"id": [None, 1], "y": ["null-right", "one-right"]}
    )

    actual = tf1.join(
        tf2,
        on="id",
        nulls_equal=True,
    ).arrange("id")

    expected = tp.TibbleFrame(
        {
            "id": [None, 1],
            "x": ["null-left", "one-left"],
            "y": ["null-right", "one-right"],
        }
    )

    assert actual.equals(expected), "null-key join failed"


def test_join_without_common_columns():
    """Raises when no join keys or common columns are available"""
    tf1 = tp.TibbleFrame({"x": [1, 2]})
    tf2 = tp.TibbleFrame({"y": [1, 2]})

    with pytest.raises(ValueError, match="No common columns"):
        tf1.join(tf2)
