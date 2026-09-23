import polars as pl

import tidypyrs as tp
from tidypyrs import f


def test_map_batches():
    actual = (
        tp.TibbleLazy(
            {
                "a": pl.int_range(-100_000, 0, eager=True),
                "b": pl.int_range(0, 100_000, eager=True),
            }
        )
        .map_batches(lambda x: 2 * x, streamable=True)
        .collect(engine="streaming")
    )

    expected = (
        tp.TibbleLazy(
            {
                "a": pl.int_range(-100_000, 0, eager=True),
                "b": pl.int_range(0, 100_000, eager=True),
            }
        )
        .select(f("a", "b") * 2)
    ).collect()

    assert actual.equals(expected)
