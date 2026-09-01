'''
uv run pytest tests/lazy/test_over.py
'''

import tidypyrs as tp
from tidypyrs import col as c

def test_filter_over_over():
    """Can filter by group"""
    tl = tp.TibbleLazy({'x': range(3), 'y': ['a', 'a', 'b']})
    actual = (
        tl.filter(c('x') <= c('x').mean(),
                  over='y')
        .arrange('y')
    )
    expected = tp.TibbleLazy({'x': [0, 2], 'y': ['a', 'b']})
    assert actual.equals(expected), "group filter failed"

def test_mutate_over():
    """Can mutate by group"""
    tl = tp.TibbleLazy({'x': range(2), 'y': ['a', 'b']})
    actual = (
        tl.mutate(avg_x = c('x').mean(),
                  over='y')
        .arrange('y')
    )
    expected = tp.TibbleLazy({'x': [0, 1], 'y': ['a', 'b'], 'avg_x': [0, 1]})
    assert actual.equals(expected), "group mutate failed"

def test_slice_over():
    """Can slice by group"""
    tl = tp.TibbleLazy({'x': range(3), 'y': ['a', 'a', 'b']})
    actual = tl.slice(0, over='y').arrange('y')
    expected = tp.TibbleLazy({'x': [0, 2], 'y': ['a', 'b']})
    assert actual.equals(expected), "group slice failed"

def test_slice_head_over():
    """Can slice_head by group"""
    tl = tp.TibbleLazy({'x': range(3), 'y': ['a', 'a', 'b']})
    actual = tl.slice_head(1, over='y').arrange('y')
    expected = tp.TibbleLazy({'x': [0, 2], 'y': ['a', 'b']})
    assert actual.equals(expected), "group slice_head failed"

def test_slice_tail_over():
    """Can slice_tail by group"""
    tl = tp.TibbleLazy({'x': range(3), 'y': ['a', 'a', 'b']})
    actual = tl.slice_tail(1, over='y').arrange('y')
    expected = tp.TibbleLazy({'x': [1, 2], 'y': ['a', 'b']})
    assert actual.equals(expected), "group slice_tail failed"
