'''
uv run pytest tests/frame/test_summarize.py
'''

import tidypyrs as tp
from tidypyrs import col as c

def test_summarize():
    """Can summarize by group"""
    tf = tp.TibbleFrame({'x': range(3)})
    actual = tf.summarize(
        avg_x = c('x').mean(),
        min_x = c('x').min()
    )
    expected = tp.TibbleFrame({
        'avg_x': [1],
        'min_x': [0]
    })
    assert actual.equals(expected), "group summarize failed"
