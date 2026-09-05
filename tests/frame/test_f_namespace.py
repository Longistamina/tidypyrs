"""
uv run pytest tests/frame/test_f_namespace.py
"""

import numpy as np
import polars as pl
import tidypyrs as tp
from tidypyrs import f
from tidypyrs.f_namespace import _defer_aware


def test_f_column_expression():
    actual = tp.TibbleFrame(x=[1, 4, 9]).mutate(root=np.sqrt(f["x"]))

    expected = tp.TibbleFrame(
        x=[1, 4, 9],
        root=[1.0, 2.0, 3.0],
    )

    assert actual.equals(expected)


def test_f_getitem_call():
    tf = tp.TibbleFrame(
        x=[1, 2, 3],
        y=[4, 5, 6],
        z=[7, 8, 9],
    )

    expected = tp.TibbleFrame(
        x=[1, 2, 3],
        y=[4, 5, 6],
    )

    assert tf.select(f["x", "y"]).equals(expected)
    assert tf.select(f[["x", "y"]]).equals(expected)
    assert tf.select(f("x", "y")).equals(expected)
    assert tf.select(f(["x", "y"])).equals(expected)


def test_f_all():
    actual = tp.TibbleFrame(
        x=[1, None],
        y=[None, 2],
    ).mutate(
        f.all().fill_null(0)
    )

    expected = tp.TibbleFrame(
        x=[1, 0],
        y=[0, 2],
    )

    assert actual.equals(expected)


def test_f_select():
    actual = tp.TibbleFrame(y=["b", "a", "b"]).mutate(y=tp.as_ordered(f.select("y")))

    assert isinstance(actual.pull("y").dtype, pl.Enum)
    assert actual.pull("y").to_list() == ["b", "a", "b"]


def test_f_select_respects_sequential_mutate():
    actual = tp.TibbleFrame(y=["b", "a", "b"]).mutate(
        copied=pl.col("y"),
        ordered=tp.as_ordered(f.select("copied")),
    )

    assert isinstance(actual.pull("ordered").dtype, pl.Enum)


@_defer_aware
def combine(a, b):
    return a, b

def test_multiple_deferred_arguments():
    operation = combine(
        f.select("x"),
        b=f.select("y"),
    )

    frame = tp.TibbleFrame(
        x=[1, 2],
        y=[3, 4],
    )

    x, y = operation.resolve(frame)

    assert x.colnames.equals(pl.Series(["x"]))
    assert y.colnames.equals(pl.Series(["y"]))


def test_f_pull():
    result = tp.TibbleFrame(
        legendary=["yes", "no", "yes"],
    ).mutate(
        f.legendary.pipe(
            tp.as_enum,
            categories=f.pull("legendary"),
        )
    )

    assert result["legendary"].dtype == pl.Enum


def test_f_pull_last_column():
    result = tp.TibbleFrame(
        x=[1, 2],
        legendary=["yes", "no"],
    ).mutate(
        f.legendary.pipe(
            tp.as_enum,
            categories=f.pull(),
        )
    )

    assert result["legendary"].dtype == pl.Enum
