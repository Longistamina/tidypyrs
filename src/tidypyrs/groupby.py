from __future__ import annotations

from datetime import timedelta
from collections.abc import Iterable
from typing import Any

from polars._utils.convert import parse_as_duration_string
from polars.dataframe.group_by import GroupBy, DynamicGroupBy
from polars.lazyframe.group_by import LazyGroupBy
from polars._plr import PyLazyGroupBy

from polars._typing import (
    ClosedInterval,
    IntoExpr,
    Label,
    StartBy,
)

from .tibble_frame import TibbleFrame, from_polars_frame
from .tibble_lazy import TibbleLazy, from_polars_lazy

# ======================================================
# GroupBy classes for TibbleFrame
# ======================================================

class TibbleGroupBy(GroupBy):
    def __init__(
        self,
        df: TibbleFrame,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        predicates: Iterable[Any] | None = None,
        **named_by: IntoExpr,
    ) -> None:
        self.df = df
        super().__init__(
            df.as_polars(),
            *by,
            maintain_order=maintain_order,
            predicates=predicates,
            **named_by
        )

    def summarize(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> TibbleFrame:
        result = super().agg(*exprs, **named_exprs)

        return from_polars_frame(result)

    summarise = summarize


class TibbleDynamicGroupBy(DynamicGroupBy):
    def __init__(
        self,
        df: TibbleFrame,
        index_column: IntoExpr,
        *,
        every: str | timedelta,
        period: str | timedelta | None,
        offset: str | timedelta | None,
        include_boundaries: bool,
        closed: ClosedInterval,
        label: Label,
        group_by: IntoExpr | Iterable[IntoExpr] | None,
        start_by: StartBy,
        predicates: Iterable[Any] | None,
    ) -> None:
        self.df = df
        super().__init__(
            df.as_polars(),
            index_column,
            every=parse_as_duration_string(every),
            period=parse_as_duration_string(period),
            offset=parse_as_duration_string(offset),
            include_boundaries=include_boundaries,
            closed=closed,
            label=label,
            group_by=group_by,
            start_by=start_by,
            predicates=predicates
        )

    def summarize(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> TibbleFrame:
        result = super().agg(*exprs, **named_exprs)

        return from_polars_frame(result)

    summarise = summarize

# ======================================================
# GroupBy class for TibbleLazy
# ======================================================

class TibbleLazyGroupBy(LazyGroupBy):
    def __init__(self, lgb: PyLazyGroupBy):
        self.lgb = lgb

    def summarize(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> TibbleLazy:
        result = super().agg(*exprs, **named_exprs)

        return from_polars_lazy(result)

    summarise = summarize
