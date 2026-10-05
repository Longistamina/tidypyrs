from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

if TYPE_CHECKING:
    from .tibble_frame import TibbleFrame
    from .tibble_lazy import TibbleLazy

__all__ = [  # noqa: RUF022
    # Experessions
    "all",
    "col",
    "concat_list",
    "element",
    "exclude",
    "lit",
    "nth",
    "select",
    "struct",
    "when",
    # Experession types
    "Expr",
    "Series",
    # selectors
    "selectors",
    # dtypes
    "Decimal",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "Boolean",
    "Binary",
    "String",
    "Array",
    "List",
    "Field",
    "Struct",
    "Time",
    "Date",
    "Datetime",
    "Duration",
    "Categories",
    "Categorical",
    "Enum",
    "Object",
    "Null",
    # sql related
    "sql",
    "sql_expr",
    "SQLContext",
    # Config
    "Config",
]

# Expressions
all = pl.all
col = pl.col
concat_list = pl.concat_list
element = pl.element
exclude = pl.exclude
lit = pl.lit
nth = pl.nth

# tp.select()
@overload
def select(*exprs: Any, eager: Literal[True] = True, **named_exprs: Any) -> "TibbleFrame": ...

@overload
def select(*exprs: Any, eager: Literal[False], **named_exprs: Any) -> "TibbleLazy": ...

def select(*exprs: Any, eager: bool = True, **named_exprs: Any) -> "TibbleFrame | TibbleLazy":
    return _wrap_result(
        pl.select(*exprs, eager=eager, **named_exprs)
    )

struct = pl.struct
when = pl.when

# Expression types
Expr = pl.Expr

class Series(pl.Series):
    def to_frame(self, name=None) -> "TibbleFrame":
        from .tibble_frame import _from_polars_frame

        out = super().to_frame(name=name)
        return _from_polars_frame(out)

# Selectors
selectors = pl.selectors

# dtypes
Decimal = pl.Decimal
Int8 = pl.Int8
Int16 = pl.Int16
Int32 = pl.Int32
Int64 = pl.Int64

UInt8 = pl.UInt8
UInt16 = pl.UInt16
UInt32 = pl.UInt32
UInt64 = pl.UInt64

Float32 = pl.Float32
Float64 = pl.Float64

Boolean = pl.Boolean
Binary = pl.Binary

String = pl.String

Array = pl.Array
List = pl.List
Field = pl.Field
Struct = pl.Struct

Time = pl.Time
Date = pl.Date
Datetime = pl.Datetime
Duration = pl.Duration

Categories = pl.Categories
Categorical = pl.Categorical
Enum = pl.Enum

Object = pl.Object

Null = pl.Null

# sql related
@overload
def sql(query: str, *, eager: Literal[True],) -> "TibbleFrame": ...

@overload
def sql(query: str, *, eager: Literal[False] = False,) -> "TibbleLazy": ...

def sql(query: str, *, eager: bool = False,) -> "TibbleFrame | TibbleLazy":
    '''
    Execute a SQL query against frames in the global namespace.

    Parameters
    ----------
    query
        SQL query to execute.

    eager
        Automatically collect the result and return a DataFrame instead of a LazyFrame.
    '''
    return _wrap_result(
        pl.sql(query=query, eager=eager)
    )

sql_expr = pl.sql_expr

class SQLContext(pl.SQLContext):
    """SQLContext that returns tidypyrs frames."""

    @overload
    def execute(self, query: str, *, eager: Literal[True]) -> "TibbleFrame": ...

    @overload
    def execute(self, query: str, *, eager: Literal[False]) -> "TibbleLazy": ...

    @overload
    def execute(self, query: str, *, eager: None = None) -> "TibbleFrame | TibbleLazy": ...

    def execute(self, query: str, *, eager: bool | None = None) -> "TibbleFrame | TibbleLazy":
        out = super().execute(query=query, eager=eager)
        return _wrap_result(out)

    @classmethod
    @overload
    def execute_global(cls, query: str, *, eager: Literal[True]) -> "TibbleFrame": ...

    @classmethod
    @overload
    def execute_global(cls, query: str, *, eager: Literal[False] = False) -> "TibbleLazy": ...

    @classmethod
    def execute_global(cls, query: str, *, eager: bool = False) -> "TibbleFrame | TibbleLazy":
        out = super().execute_global(
            query=query,
            eager=eager,
        )
        return _wrap_result(out)
# Config
Config = pl.Config

# wrapt output to TibbleFrame or TibbleLazy
def _wrap_result(out: pl.DataFrame | pl.LazyFrame,) -> "TibbleFrame | TibbleLazy":
    if isinstance(out, pl.LazyFrame):
        from .tibble_lazy import as_tl
        return as_tl(out)
    from .tibble_frame import as_tf
    return as_tf(out)
