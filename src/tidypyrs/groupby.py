from __future__ import annotations

from collections.abc import Callable

import polars as pl

from .f_namespace import _Deferred


def _call_group_function(
    function: Callable,
    group: pl.DataFrame,
) -> pl.DataFrame:
    """
    Call a group UDF with a TibbleFrame.

    Polars passes a native DataFrame to map_groups(), so convert it
    before calling the user function. Convert the returned TibbleFrame
    back to a native DataFrame for Polars.
    """
    # Local import avoids the circular import:
    #
    # tibble_frame -> groupby -> tibble_frame
    from .tibble_frame import TibbleFrame, _from_polars_frame

    tibble_group = _from_polars_frame(group)
    result = function(tibble_group)

    if isinstance(result, TibbleFrame):
        return result.as_polars()

    if isinstance(result, pl.DataFrame):
        return result

    raise TypeError(
        "`map_groups()` functions must return a "
        "TibbleFrame or Polars DataFrame, "
        f"not {type(result).__name__}"
    )


# ======================================================
# GroupBy class for TibbleFrame
# ======================================================


class TibbleGroupBy:
    def __init__(self, group_by, wrap):
        self._group_by = group_by
        self._wrap = wrap

    def agg(self, *exprs, **named_exprs):
        return self._wrap(
            self._group_by.agg(*exprs, **named_exprs)
        )

    summarize = agg
    summarise = agg

    def map_groups(self, function):
        out = self._group_by.map_groups(
            lambda group: _call_group_function(
                function,
                group,
            )
        )

        return self._wrap(out)


# ======================================================
# GroupBy class for TibbleLazy
# ======================================================


class TibbleLazyGroupBy:
    def __init__(self, group_by, wrap, frame):
        self._group_by = group_by
        self._wrap = wrap
        self._frame = frame

    def agg(self, *exprs, **named_exprs):
        return self._wrap(
            self._group_by.agg(*exprs, **named_exprs)
        )

    summarize = agg
    summarise = agg

    def map_groups(self, function, *, schema):
        if isinstance(schema, _Deferred):
            schema = schema.resolve(self._frame)

        out = self._group_by.map_groups(
            lambda group: _call_group_function(
                function,
                group,
            ),
            schema=schema,
        )

        return self._wrap(out)
