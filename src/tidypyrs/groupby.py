from __future__ import annotations
from .f_namespace import _Deferred

# ======================================================
# GroupBy classes for TibbleFrame
# ======================================================


class TibbleGroupBy:
    def __init__(self, group_by, wrap):
        self._group_by = group_by
        self._wrap = wrap

    def agg(self, *exprs, **named_exprs):
        return self._wrap(self._group_by.agg(*exprs, **named_exprs))

    summarize = agg
    summarise = agg

    def map_groups(self, function):
        return self._wrap(self._group_by.map_groups(function))


# ======================================================
# GroupBy class for TibbleLazy
# ======================================================


class TibbleLazyGroupBy:
    def __init__(self, group_by, wrap, frame):
        self._group_by = group_by
        self._wrap = wrap
        self._frame = frame

    def agg(self, *exprs, **named_exprs):
        return self._wrap(self._group_by.agg(*exprs, **named_exprs))

    summarize = agg
    summarise = agg

    def map_groups(self, function, *, schema):
        if isinstance(schema, _Deferred):
            schema = schema.resolve(self._frame)
        return self._wrap(self._group_by.map_groups(function, schema=schema))
