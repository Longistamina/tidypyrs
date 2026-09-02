import polars.selectors as cs
import re

__all__ = ["contains", "ends_with", "everything", "starts_with", "where"]

def contains(match, ignore_case=True):
    """
    Select columns whose names contain a literal string.

    Parameters
    ----------
    match : str
        Literal substring to find in column names.

    ignore_case : bool
        If True, ignore case when matching names.
    """
    if ignore_case:
        return cs.matches(rf"(?i){re.escape(match)}")

    return cs.contains(match)

def starts_with(match, ignore_case=True):
    """
    Starts with a prefix

    Parameters
    ----------
    match : str
        String to match columns
    ignore_case : bool
        If TRUE, the default, ignores case when matching names.

    Examples
    --------
    >>> tf = tp.TibbleFrame{'a': range(3), 'add': range(3), 'sub': ['a', 'a', 'b']})
    >>> tf.select(tp.starts_with('a'))
    """
    if ignore_case == True:
        out = cs.matches(f"^(?i){match}.*$")
    else:
        out = cs.starts_with(match)
    return out

def ends_with(match, ignore_case=True):
    """
    Ends with a suffix

    Parameters
    ----------
    match : str
        String to match columns

    ignore_case : bool
        If TRUE, the default, ignores case when matching names.

    Examples
    --------
    >>> tf = tp.TibbleFrame{'a': range(3), 'b_code': range(3), 'c_code': ['a', 'a', 'b']})
    >>> tf.select(tp.ends_with('code'))
    """
    if ignore_case == True:
        out = cs.matches(f"^.*(?i){match}$")
    else:
        out = cs.ends_with(match)
    return out

def everything():
    """
    Selects all columns

    Examples
    --------
    >>> tf = tp.TibbleFrame{'a': range(3), 'b': range(3), 'c': ['a', 'a', 'b']})
    >>> tf.select(tp.everything())
    """
    return cs.all()

_col_types = {
    "date": cs.date(),
    "datetime": cs.datetime(),
    "temporal": cs.temporal(),
    "float": cs.float(),
    "integer": cs.integer(),
    "numeric": cs.numeric(),
    "string": cs.string(),
    "categorical": cs.categorical(),
    "factor": cs.categorical(),
    "ordered": cs.enum(),
    "enum": cs.enum()
}

def where(col_type):
    """
    Select columns by type using a string

    Options:
        date, datetime, float, integer,
        numeric, string

    Examples
    --------
    >>> tf.select(tp.where("integer"))
    """
    out = _col_types[col_type]
    return out
