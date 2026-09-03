import polars as pl  # noqa: I001
import functools as ft
from .utils import _as_list, _col_expr

__all__ = [  # noqa: RUF022
    "str_length",
    #####################
    "str_to_lower",
    "str_to_upper",
    #####################
    "str_concat",
    "str_paste",
    "str_paste0",
    #####################
    "str_detect",
    "str_starts",
    "str_ends",
    #####################
    "str_replace",
    "str_replace_all",
    #####################
    "str_extract",
    "str_sub",
    #####################
    "str_remove_all",
    "str_remove",
    "str_trim",
]


def str_length(string):
    """
    Length of a string

    Parameters
    ----------
    string : str
        Input series to operate on

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_length(tp.col('name')))
    """
    string = _col_expr(string)
    return string.str.len_bytes()


##------------------------------------##


def str_to_lower(string):
    """
    Convert case of a string

    Parameters
    ----------
    string : str
        Convert case of this string

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_to_lower(tp.col('name')))
    """
    string = _col_expr(string)
    return string.str.to_lowercase()


def str_to_upper(string):
    """
    Convert case of a string

    Parameters
    ----------
    string : str
        Convert case of this string

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_to_upper(tp.col('name')))
    """
    string = _col_expr(string)
    return string.str.to_uppercase()


##------------------------------------##


def str_paste(*args, sep=" "):
    """
    Concatenate strings together

    Parameters
    ----------
    args : Expr, str
        Columns and or strings to concatenate

    Examples
    --------
    >>> tf = tp.TibbleFrame(x = ['a', 'b', 'c'])
    >>> tf.mutate(x_end = tp.paste(tp.col('x'), 'end', sep = '_'))
    """
    args = _as_list(args)
    args = [pl.lit(arg) if not isinstance(arg, pl.Expr) else arg for arg in args]  # [pl.lit(arg), pl.lit(arg), pl.lit(arg), ...]
    curlies = ["{}"] * len(args)  # ['{}', '{}', '{}', '{}', ...]
    string_format = sep.join(curlies)  # "{}sep{}sep{}sep{}sep{}...{}"
    return pl.format(string_format, *args)  # "{arg}sep{arg}sep{arg}...{arg}"


def str_paste0(*args):
    """
    Concatenate strings together with no separator

    Parameters
    ----------
    args : Expr, str
        Columns and or strings to concatenate

    Examples
    --------
    >>> tf = tp.TibbleFrame(x = ['a', 'b', 'c'])
    >>> tf.mutate(xend = tp.paste0(tp.col('x'), 'end'))
    """
    return str_paste(*args, sep="")


def str_concat(*args, sep=""):
    """
    Concatenate strings together

    Parameters
    ----------
    args : Expr, str
        Columns and/or strings to concatenate

    Examples
    --------
    >>> tf = tp.TibbleFrame(x = ['a', 'b', 'c'])
    >>> tf.mutate(x_end = str_c(tp.col('x'), 'end', sep = '_'))
    """
    return str_paste(*args, sep=sep)


##------------------------------------##


def str_detect(string, pattern, negate=False):
    """
    Detect the presence or absence of a pattern in a string

    Parameters
    ----------
    string : str
        Input series to operate on
    pattern : str
        Pattern to look for
    negate : bool
        If True, return non-matching elements

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_detect('name', 'a'))
    >>> tf.mutate(x = str_detect('name', ['a', 'e']))
    """
    if isinstance(pattern, str):  # "a" -> ["a"]
        pattern = [pattern]

    string = _col_expr(string)  # "col_name" -> pl.lit("col_name")

    exprs = (string.str.contains(p) for p in pattern)  # lazy generator of (True, False, True, False, ...)
    exprs = ft.reduce(lambda a, b: a & b, exprs)  # True if all are True, else False
    if negate:
        exprs = exprs.not_()

    return exprs


def str_starts(string, pattern, negate=False):
    """
    Detect the presence or absence of a pattern at the beginning of a string.

    Parameters
    ----------
    string : Expr
        Column to operate on
    pattern : str
        Pattern to look for
    negate : bool
        If True, return non-matching elements

    Examples
    --------
    >>> tf = tp.TibbleFrame(words = ['apple', 'bear', 'amazing'])
    >>> tf.filter(tp.str_starts(tp.col('words'), 'a'))
    """
    pattern = "^" + pattern
    return str_detect(string, pattern, negate)


def str_ends(string, pattern, negate=False):
    """
    Detect the presence or absence of a pattern at the end of a string.

    Parameters
    ----------
    string : Expr
        Column to operate on
    pattern : str
        Pattern to look for
    negate : bool
        If True, return non-matching elements

    Examples
    --------
    >>> tf = tp.TibbleFrame(words = ['apple', 'bear', 'amazing'])
    >>> tf.filter(tp.str_ends(tp.col('words'), 'ing'))
    """
    pattern = pattern + "$"
    return str_detect(string, pattern, negate)


##------------------------------------##


def str_replace(string, pattern, replacement):
    """
    Replaces the first matched patterns in a string

    Parameters
    ----------
    string : str
        Input series to operate on
    pattern : str
        Pattern to look for
    replacement : str
        String that replaces anything that matches the pattern

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_replace(tp.col('name'), 'a', 'A'))
    """
    string = _col_expr(string)
    return string.str.replace(pattern, replacement)


def str_replace_all(string, pattern, replacement):
    """
    Replaces all matched patterns in a string

    Parameters
    ----------
    string : str
        Input series to operate on
    pattern : str
        Pattern to look for
    replacement : str
        String that replaces anything that matches the pattern

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_replace_all(tp.col('name'), 'a', 'A'))
    """
    string = _col_expr(string)
    return string.str.replace_all(pattern, replacement)


##------------------------------------##


def str_sub(string, start=0, end=None):
    """
    Extract portion of string based on start and end indices

    Parameters
    ----------
    string : str
        Input series to operate on
    start : int
        First position index of the character to return
    end : int
        Last position index of the character to return

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_sub(tp.col('name'), 0, 3))
    """
    string = _col_expr(string)
    return string.str.slice(start, end)


def str_extract(string, pattern):
    """
    Extract the target capture group from provided patterns

    Parameters
    ----------
    string : str
        Input series to operate on
    pattern : str
        Pattern to look for

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_extract(tp.col('name'), 'e'))
    """
    string = _col_expr(string)
    return string.str.extract(pattern, 0)


##------------------------------------##


def str_remove(string, pattern):
    """
    Removes the first matched patterns in a string

    Parameters
    ----------
    string : str
        Input series to operate on
    pattern : str
        Pattern to look for

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_remove(col('name'), 'a'))
    """
    return str_replace(string, pattern, "")


def str_remove_all(string, pattern):
    """
    Removes all matched patterns in a string

    Parameters
    ----------
    string : str
        Input series to operate on
    pattern : str
        Pattern to look for

    Examples
    --------
    >>> tf = tp.TibbleFrame(name = ['apple', 'banana', 'pear', 'grape'])
    >>> tf.mutate(x = str_remove_all(col('name'), 'a'))
    """
    return str_replace_all(string, pattern, "")


##------------------------------------##


def str_trim(string, side="both"):
    """
    Trim whitespace

    Parameters
    ----------
    string : Expr, Series
        Column or series to operate on
    side : str
        One of:
            * "both"
            * "left"
            * "right"

    Examples
    --------
    >>> tf = tp.TibbleFrame(x = [' a ', ' b ', ' c '])
    >>> tf.mutate(x = tp.str_trim(col('x')))
    """
    string = _col_expr(string)
    if side == "both":
        out = _str_trim_right(_str_trim_left(string))
    elif side == "left":
        out = _str_trim_left(string)
    elif side == "right":
        out = _str_trim_right(string)
    else:
        raise ValueError("side must be one of 'both', 'left', or 'right'")
    return out


def _str_trim_left(x):
    """
    Remove leading whitespace.
    """
    return x.str.replace(r"^\s*", "")


def _str_trim_right(x):
    """
    Remove trailing whitespace.
    """
    return x.str.replace(r"[ \t]+$", "")
