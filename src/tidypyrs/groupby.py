from __future__ import annotations

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


# ======================================================
# GroupBy class for TibbleLazy
# ======================================================


class TibbleLazyGroupBy:
    def __init__(self, group_by, wrap):
        self._group_by = group_by
        self._wrap = wrap

    def agg(self, *exprs, **named_exprs):
        return self._wrap(self._group_by.agg(*exprs, **named_exprs))

    summarize = agg
    summarise = agg
