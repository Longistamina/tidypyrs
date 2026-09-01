import polars as pl
from collections.abc import Callable
from typing import Any

# ======================================
# define _Deferred class
# ======================================

class _Deferred:
    """An operation waiting for a DataFrame or LazyFrame."""

    __slots__ = ("_resolver",)

    def __init__(self, resolver: Callable[[Any], Any]): # `resolver` takes one argument of any type [Any], and return one value of any type Any
        self._resolver = resolver

    def resolve(self, frame):
        return self._resolver(frame)

    def map(self, function):
        return _Deferred(
            lambda frame: function(self.resolve(frame))
        )

    def alias(self, name):
        return self.map(lambda expression: expression.alias(name))

# =============================================
# define _FrameReference for ``f`` namespace
# =============================================

class _FrameReference:
    __slots__ = ()

    def __getitem__(self, name: str) -> pl.Expr:
        """
        Return a column expression.

        -----------------------------
        Example:
            f["a"] -> pl.col("a")
        """
        return pl.col(name)

    def __getattr__(self, name: str) -> pl.Expr:
        """
        Allow f.column_name syntax.

        --------------------------------
        Example:
            f.a -> pl.col("a")
        """
        if name.startswith("_"):
            raise AttributeError(name)

        return pl.col(name)

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
                y = tp.as_ordered(f.select("y"))
            )
        """
        return _Deferred(
            lambda frame: frame.select(*exprs, **named_exprs)
        )

    def __repr__(self):
        return "f"

f = _FrameReference()

__all__ = ["f"]

'''
# =======================================================
# Explanation
# =======================================================

##-----------------------##
## The goal of this file ##
##-----------------------##

'''
