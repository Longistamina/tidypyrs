'''
uv run pytest tests/lazy/test_groupby.py
'''

from datetime import datetime
import tidypyrs as tp
from tidypyrs import col as c

def test_group_by_summarize():
    """Can group by one column and summarize."""
    tl = tp.TibbleLazy({
        'group': ['a', 'a', 'b'],
        'value': [1, 2, 3],
    })

    actual = (
        tl
        .group_by('group')
        .summarize(
            total=c('value').sum(),
            average=c('value').mean(),
        )
        .arrange('group')
    )

    expected = tp.TibbleLazy({
        'group': ['a', 'b'],
        'total': [3, 3],
        'average': [1.5, 3.0],
    })

    assert isinstance(actual, tp.TibbleLazy)
    assert actual.equals(expected), "single-column group_by failed"


def test_group_by_summarise_alias():
    """The British spelling summarise is supported."""
    tl = tp.TibbleLazy({
        'group': ['a', 'a', 'b'],
        'value': [1, 2, 3],
    })

    actual = (
        tl
        .group_by('group')
        .summarise(total=c('value').sum())
        .arrange('group')
    )

    expected = tp.TibbleLazy({
        'group': ['a', 'b'],
        'total': [3, 3],
    })

    assert isinstance(actual, tp.TibbleLazy)
    assert actual.equals(expected), "group_by.summarise alias failed"


def test_group_by_agg():
    """Native agg syntax returns a TibbleLazy."""
    tl = tp.TibbleLazy({
        'group': ['a', 'a', 'b'],
        'value': [1, 2, 3],
    })

    actual = (
        tl
        .group_by('group')
        .agg(total=c('value').sum())
        .arrange('group')
    )

    expected = tp.TibbleLazy({
        'group': ['a', 'b'],
        'total': [3, 3],
    })

    assert isinstance(actual, tp.TibbleLazy)
    assert actual.equals(expected), "group_by.agg failed"


def test_group_by_multiple_columns():
    """Can group by multiple columns."""
    tl = tp.TibbleLazy({
        'group': ['a', 'a', 'a', 'b'],
        'kind': ['x', 'x', 'y', 'x'],
        'value': [1, 2, 3, 4],
    })

    actual = (
        tl
        .group_by('group', 'kind')
        .summarize(total=c('value').sum())
        .arrange('group', 'kind')
    )

    expected = tp.TibbleLazy({
        'group': ['a', 'a', 'b'],
        'kind': ['x', 'y', 'x'],
        'total': [3, 3, 4],
    })

    assert actual.equals(expected), "multiple-column group_by failed"


def test_group_by_maintain_order():
    """maintain_order preserves first-seen group order."""
    tl = tp.TibbleLazy({
        'group': ['b', 'a', 'b', 'c'],
        'value': [1, 2, 3, 4],
    })

    actual = (
        tl
        .group_by('group', maintain_order=True)
        .summarize(total=c('value').sum())
    )

    expected = tp.TibbleLazy({
        'group': ['b', 'a', 'c'],
        'total': [4, 2, 4],
    })

    assert actual.equals(expected), "group_by maintain_order failed"


def test_group_by_dynamic():
    """Can aggregate rows in weekly dynamic windows."""
    tl = tp.TibbleLazy({
        'date': [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 8),
            datetime(2024, 1, 9),
        ],
        'value': [1, 2, 3, 4],
    })

    actual = (
        tl
        .group_by_dynamic('date', every='1w')
        .summarize(total=c('value').sum())
        .arrange('date')
    )

    expected = tp.TibbleLazy({
        'date': [
            datetime(2024, 1, 1),
            datetime(2024, 1, 8),
        ],
        'total': [3, 7],
    })

    assert isinstance(actual, tp.TibbleLazy)
    assert actual.equals(expected), "group_by_dynamic failed"


def test_group_by_dynamic_with_group_by():
    """Dynamic windows can also partition by another column."""
    tl = tp.TibbleLazy({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'date': [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 8),
            datetime(2024, 1, 1),
            datetime(2024, 1, 8),
        ],
        'value': [1, 2, 3, 4, 5],
    })

    actual = (
        tl
        .group_by_dynamic(
            'date',
            every='1w',
            group_by='group',
        )
        .summarise(total=c('value').sum())
        .arrange('group', 'date')
    )

    expected = tp.TibbleLazy({
        'group': ['a', 'a', 'b', 'b'],
        'date': [
            datetime(2024, 1, 1),
            datetime(2024, 1, 8),
            datetime(2024, 1, 1),
            datetime(2024, 1, 8),
        ],
        'total': [3, 3, 4, 5],
    })

    assert isinstance(actual, tp.TibbleLazy)
    assert actual.equals(expected), "grouped group_by_dynamic failed"
