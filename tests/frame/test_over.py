"""
uv run pytest tests/frame/test_over.py
"""

import tidypyrs as tp
from tidypyrs import col as c


def test_filter_over():
    """Can filter by group"""
    tf = tp.TibbleFrame({"x": range(3), "y": ["a", "a", "b"]})
    actual = tf.filter(c("x") <= c("x").mean(), over="y").arrange("y")
    expected = tp.TibbleFrame({"x": [0, 2], "y": ["a", "b"]})
    assert actual.equals(expected), "group filter failed"


def test_mutate_over():
    """Can mutate by group"""
    tf = tp.TibbleFrame({"x": range(2), "y": ["a", "b"]})
    actual = tf.mutate(avg_x=c("x").mean(), over="y").arrange("y")
    expected = tp.TibbleFrame({"x": [0, 1], "y": ["a", "b"], "avg_x": [0, 1]})
    assert actual.equals(expected), "group mutate failed"


def test_slice_over():
    """Can slice by group"""
    tf = tp.TibbleFrame({"x": range(3), "y": ["a", "a", "b"]})
    actual = tf.slice(0, over="y").arrange("y")
    expected = tp.TibbleFrame({"x": [0, 2], "y": ["a", "b"]})
    assert actual.equals(expected), "group slice failed"


def test_slice_head_over():
    """Can slice_head by group"""
    tf = tp.TibbleFrame({"x": range(3), "y": ["a", "a", "b"]})
    actual = tf.slice_head(1, over="y").arrange("y")
    expected = tp.TibbleFrame({"x": [0, 2], "y": ["a", "b"]})
    assert actual.equals(expected), "group slice_head failed"


def test_slice_tail_over():
    """Can slice_tail by group"""
    tf = tp.TibbleFrame({"x": range(3), "y": ["a", "a", "b"]})
    actual = tf.slice_tail(1, over="y").arrange("y")
    expected = tp.TibbleFrame({"x": [1, 2], "y": ["a", "b"]})
    assert actual.equals(expected), "group slice_tail failed"
