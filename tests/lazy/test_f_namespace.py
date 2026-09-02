"""
uv run pytest tests/lazy/test_f_namespace.py
"""

import numpy as np
import polars as pl
import tidypyrs as tp
from tidypyrs import f


def test_f_column_expression():
    actual = tp.TibbleLazy(x=[1, 4, 9]).mutate(root=np.sqrt(f["x"]))

    expected = tp.TibbleLazy(
        x=[1, 4, 9],
        root=[1.0, 2.0, 3.0],
    )

    assert actual.equals(expected)


def test_f_getitem_call():
    tl = tp.TibbleLazy(
        x=[1, 2, 3],
        y=[4, 5, 6],
        z=[7, 8, 9],
    )

    expected = tp.TibbleLazy(
        x=[1, 2, 3],
        y=[4, 5, 6],
    )

    assert tl.select(f["x", "y"]).equals(expected)
    assert tl.select(f[["x", "y"]]).equals(expected)
    assert tl.select(f("x", "y")).equals(expected)
    assert tl.select(f(["x", "y"])).equals(expected)


def test_f_select():
    actual = tp.TibbleLazy(y=["b", "a", "b"]).mutate(y=tp.as_ordered(f.select("y")))

    assert isinstance(actual.pull("y").dtype, pl.Enum)
    assert actual.pull("y").to_list() == ["b", "a", "b"]


def test_f_select_respects_sequential_mutate():
    actual = tp.TibbleLazy(y=["b", "a", "b"]).mutate(
        copied=pl.col("y"),
        ordered=tp.as_ordered(f.select("copied")),
    )

    assert isinstance(actual.pull("ordered").dtype, pl.Enum)
