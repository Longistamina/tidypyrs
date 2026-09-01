import polars as pl
import polars.selectors as cs
from operator import not_
from itertools import chain
from .f_namespace import _Deferred

__all__ = []

# =======================================
# Safe length utility
# =======================================

def _safe_len(x):
    if not_(hasattr(x, "__len__")):
        return 0
    else:
        return len(x)

# =======================================
# Type checking utilities
# =======================================

def _is_boolean(x):
    return isinstance(x, bool)

def _is_integer(x):
    return isinstance(x, int)

def _is_float(x):
    return isinstance(x, float)

def _is_string(x):
    return isinstance(x, str)

def _is_constant(x):
    return _is_boolean(x) | _is_integer(x) | _is_float(x) | _is_string(x)

def _is_list(x):
    return isinstance(x, list)

def _is_tuple(x):
    return isinstance(x, tuple)

def _is_iterable(x):
    return hasattr(x, '__iter__') & not_(_is_string(x))

def _is_series(x):
    return isinstance(x, pl.Series)

def _is_expr(x):
    return isinstance(x, pl.Expr)

def _is_type(x):  # Check single literal Polars value like pl.Int8 (``type(pl.Int8).__name__`` will return "DataTypeClass")
    return type(x).__name__ == 'DataTypeClass'

# ==========================================================================================
# Check if user uses ``over`` parameter in any function that supports this parameter
# And convert ``expr`` to ``expr.over(groups)``
# ==========================================================================================

def _uses_over(over):
    return over is not None and len(_as_list(over)) > 0

def _over_exprs(exprs, over):
    if not _uses_over(over):
        return exprs

    groups = _as_list(over)
    return [expr.over(groups) for expr in exprs]

# =======================================
# List related utilities
# =======================================

def _list_flatten(l):
    l = [x if isinstance(x, list) else [x] for x in l] # create nested list
    return list(chain.from_iterable(l)) # [[1, 2, 3], [4], [5, 6]] -> [1, 2, 3, 4, 5, 6]

def _as_list(x):
    if x is None:
        return []
    if _is_type(x):
        return [x]
    if _is_series(x):
        return x.to_list()
    if isinstance(x, (list, tuple)):
        return _list_flatten([
            value.to_list() if _is_series(value) else value
            for value in x
        ])
    return [x]

def _repeat(x, times):
    if not_(_is_list(x)): # ensure x is a list
        x = [x]
    return x * times

# ===================================================
# Convert a string to a Polars literal expression
# ===================================================

def _str_to_lit(x):
    if _is_string(x):
        x = pl.lit(x)
    return x

# ========================================================
# Convert a scalar value to a Polars literal expression
# ========================================================

def _lit_expr(x):
    # Deferred expressions must be resolved by _mutate_cols().
    if isinstance(x, _Deferred):
        return x

    if not _is_expr(x):
        x = pl.lit(x)

    return x

# ======================================================
# Mutate columns with given dataframe and expressions
# ======================================================

def _mutate_cols(frame, exprs):
    for expr in exprs:
        if isinstance(expr, _Deferred):
            expr = expr.resolve(frame)

        frame = frame.with_columns(expr)

    return frame

# ======================================================
# Column expression related utilities
# ======================================================

# Convert input x into a column expression
def _col_expr(x):
    if _is_expr(x) | _is_series(x) | cs.is_selector(x):
        return x
    elif _is_string(x) | _is_type(x):
        return pl.col(x)
    else:
       raise ValueError("Invalid input for column selection")

#  Wrap all str inputs in col()
def _col_exprs(x):
    if _is_list(x) | _is_series(x):
        return [_col_expr(val) for val in x]
    else:
        return [_col_expr(x)]

# Convert kwargs to col() expressions with alias
def _kwargs_as_exprs(kwargs):
    return [_lit_expr(expr).alias(key) for key, expr in kwargs.items()]
