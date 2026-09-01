'''
uv run pytest tests/lazy/test_summarize.py
'''

import tidypyrs as tp
from tidypyrs import col as c

def test_summarize():
    """Can summarize by group"""
    tl = tp.TibbleLazy({'x': range(3)})
    actual = tl.summarize(
        avg_x = c('x').mean(),
        min_x = c('x').min()
    )
    expected = tp.TibbleLazy({
        'avg_x': [1],
        'min_x': [0]
    })
    assert actual.equals(expected), "group summarize failed"
