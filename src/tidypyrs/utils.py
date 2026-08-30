import polars as pl
import polars.selectors as cs
from operator import not_
from itertools import chain

__all__ = []

# =======================================
# Safe length utility
# =======================================

def _safe_len(x):
    if x == None:
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
# Check if user uses ``by`` parameter in any function that supports this parameter
# ==========================================================================================

def _uses_by(by):
    if _is_expr(by) | _is_string(by):
        return True
    elif isinstance(by, list):
        # Allow passing an empty list to `by`
        if _safe_len(by) == 0:
            return False
        else:
            return True
    else:
        return False

# =======================================
# List related utilities
# =======================================

def _list_flatten(l):
    l = [x if isinstance(x, list) else [x] for x in l] # create nested list
    return list(chain.from_iterable(l)) # [[1, 2, 3], [4], [5, 6]] -> [1, 2, 3, 4, 5, 6]

def _as_list(x):
    if _is_type(x): # Convert single literal Polars value like ``pl.Int8`` into a single-element list ``[pl.Int8]``
        out = [x]
    elif _safe_len(x) == 0:
        out = []
    elif _is_series(x):
        out = x.to_list() # use ``to_list()`` method of pl.Series to convert it into a Python list
    elif _is_tuple(x):
        # Helpful to convert args to a list
        out = [val.to_list() if _is_series(val) else val for val in x]
        out = _list_flatten(x)
    elif _is_list(x):
        out = _list_flatten(x)
    else:
        out = [x] # 3 -> [3]
    return out

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
    if not_(_is_expr(x)):
        x = pl.lit(x)
    return x

# ======================================================
# Mutate columns with given dataframe and expressions
# ======================================================

def _mutate_cols(df, exprs):
    for expr in exprs:
        df = df.with_columns(expr)
    return df

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
    return [_lit_expr(expr).alias(key) for key, expr in kwargs.item()]
