'''Detailed explanations of this file is in `notes/f_namespace_explain.md`'''

import polars as pl
from collections.abc import Callable
from typing import Any
from functools import wraps

# ======================================
# define _Deferred class
# ======================================


class _Deferred:
    """An operation waiting for a DataFrame or LazyFrame."""

    __slots__ = ("_resolver",)

    def __init__(self, resolver: Callable[[Any], Any]):  # `resolver` function takes one argument of any type [Any], and return one value of any type Any
        self._resolver = resolver

    def resolve(self, frame):
        return self._resolver(frame)

    def map(self, function):
        return _Deferred(lambda frame: function(self.resolve(frame)))

    def alias(self, name):
        return self.map(lambda expression: expression.alias(name))


# =============================================
# @_deferred_aware decorator
# =============================================

def _defer_aware(function):
    @wraps(function)
    def wrapper(x, *args, **kwargs):
        if isinstance(x, _Deferred):
            return x.map(
                lambda resolved: function(
                    resolved,
                    *args,
                    **kwargs,
                )
            )

        return function(x, *args, **kwargs)

    return wrapper

# =============================================
# define _FrameReference for ``f`` namespace
# =============================================


class _FrameReference:
    __slots__ = ()

    def __getitem__(self, *names) -> pl.Expr:
        """
        Return a column expression.

        -----------------------------
        Examples:
            f["a"] -> pl.col("a")
            f["a", "b"] -> pl.col(["a", "b"])
        """
        return pl.col(*names)

    def __getattr__(self, name) -> pl.Expr:
        """
        Allow f.column_name syntax.

        --------------------------------
        Example:
            f.a -> pl.col("a")
        """
        if name.startswith("_"):
            raise AttributeError(name)

        return pl.col(name)

    def __call__(self, *names) -> pl.Expr:
        """
        Allow f("col") syntax.

        --------------------------------
        Examples:
            f("a") -> pl.col("a")
            f("a", "b", "c") -> pl.col("a", "b", "c")
        """
        return pl.col(*names)

    def select(self, *exprs, **named_exprs) -> _Deferred:
        """
        Select from the frame currently executing the verb.
        Help eliminate pipe functions

        ------------------------------------------------------
        Example:
            tl = tp.TibbleLazy(
                x = range(0, 10, 2),
                y = ["a", "b", "b", "c", "a"]
            ).mutate(
                y = tp.as_enum(f.select("y"))
            )
        """
        return _Deferred(lambda frame: frame.select(*exprs, **named_exprs))

    sl = select  # Allows `f.sl("a")`

    def __repr__(self):
        return "f"


f = _FrameReference()

__all__ = ["f"]
