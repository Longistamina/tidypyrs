import polars as pl
import tidypyrs as tp


def test_eager_public_api():
    tf = tp.TibbleFrame(
        group=["a", "a", "b"],
        value=[1, 2, 3],
    )

    result = tf.mutate(
        group_mean=tp.mean("value"),
        over="group",
    )

    assert isinstance(result, tp.TibbleFrame)
    assert result.colnames.to_list() == ["group", "value", "group_mean"]


def test_lazy_public_api():
    tl = tp.TibbleLazy(
        group=["a", "a", "b"],
        value=[1, 2, 3],
    )

    result = tl.mutate(
        group_mean=tp.mean("value"),
        over="group",
    )

    print(result.collect())

    assert isinstance(result.colnames, pl.Series)
    assert len(result.colnames) > 0
    assert isinstance(result, tp.TibbleLazy)
    assert isinstance(result.as_polars(), pl.LazyFrame)
    assert isinstance(result.collect(), tp.TibbleFrame)


test_lazy_public_api()


def test_grouped_summary_return_types():
    tf = tp.TibbleFrame(group=["a", "a", "b"], value=[1, 2, 3])
    tl = tf.lazy()

    eager = tf.group_by("group").summarize(mean_value=tp.mean("value"))
    lazy = tl.group_by("group").summarize(mean_value=tp.mean("value"))

    assert isinstance(eager, tp.TibbleFrame)
    assert isinstance(lazy, tp.TibbleLazy)


def test_every_export_exists():
    missing = [name for name in tp.__all__ if not hasattr(tp, name)]

    assert missing == []


def test_no_duplicate_exports():
    assert len(tp.__all__) == len(set(tp.__all__))


"""
uv run python -c "import tidypyrs as tp; print(tp.__all__)"
uv run pytest tests/test_public_apis.py
uv run ruff check src tests
uv run pyright
uv build
"""
