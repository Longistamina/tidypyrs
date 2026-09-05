'''Detailed explanations of this file is in `notes/f_namespace_explain.md`'''

from collections.abc import Callable
from functools import wraps
from typing import Any

import polars as pl

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
    def wrapper(*args, **kwargs):
        has_deferred = (
            any(isinstance(arg, _Deferred) for arg in args)
            or any(
                isinstance(value, _Deferred)
                for value in kwargs.values()
            )
        )

        if not has_deferred:
            return function(*args, **kwargs)

        def resolver(frame):
            resolved_args = tuple(
                arg.resolve(frame)
                if isinstance(arg, _Deferred)
                else arg
                for arg in args
            )

            resolved_kwargs = {
                key: (
                    value.resolve(frame)
                    if isinstance(value, _Deferred)
                    else value
                )
                for key, value in kwargs.items()
            }

            return function(
                *resolved_args,
                **resolved_kwargs,
            )

        return _Deferred(resolver)

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

    def all(self):
        """
        Select all columns as a Polars expression.

        --------------------------------
        f.all() -> pl.all()
        """
        return pl.all()


    def pull(self, var=None) -> _Deferred:
        """
        Defer extracting a column from the current frame as a Series.

        Parameters
        ----------
        var : str, optional
            Column to extract. If omitted, extract the last column
            of the frame available when this operation is resolved.

        Returns
        -------
        _Deferred
            An operation that resolves to a Polars Series.

        Notes
        -----
        Resolving this operation against a TibbleLazy collects the
        selected column.
        """
        from .funs import from_polars
        return _Deferred(lambda frame: from_polars(frame).pull(var))

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
