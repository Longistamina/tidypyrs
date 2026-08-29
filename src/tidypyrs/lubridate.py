import polars as pl
from .utils import _col_expr

__all__ = [
    "as_date",
    "as_datetime",
    "hour",
    "make_date",
    "make_datetime",
    "mday",
    "minute",
    "month",
    "quarter",
    "dt_round",
    "second",
    "wday",
    "week",
    "yday",
    "year"
]

def as_date(x, format = None):
    """
    Convert a string to a Date

    Parameters
    ----------
    x : Expr, Series
        Column to operate on
    fmt: str
        "yyyy-mm-dd"

    Examples
    --------
    >>> df = tp.tibble(x = ['2021-01-01', '2021-10-01'])
    >>> df.mutate(date_x = tp.as_date(tp.col('x')))
    """
    x = _col_expr(x)
    return x.str.strptime(pl.Date, format = format)

def as_datetime(x, format = None):
    """
    Convert a string to a Datetime

    Parameters
    ----------
    x : Expr, Series
        Column to operate on
    fmt: str
        "yyyy-mm-dd"

    Examples
    --------
    >>> df = tp.tibble(x = ['2021-01-01', '2021-10-01'])
    >>> df.mutate(date_x = tp.as_datetime(tp.col('x')))
    """
    x = _col_expr(x)
    return x.str.strptime(pl.Datetime, format = format)

def hour(x):
    """
    Extract the hour from a datetime

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(hour = tp.as_hour(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.hour()

def mday(x):
    """
    Extract the month day from a date from 1 to 31.

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(monthday = tp.mday(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.day()

def make_date(year = 1970, month = 1, day = 1):
    """
    Create a date object

    Parameters
    ----------
    year : Expr, str, int
        Column or literal
    month : Expr, str, int
        Column or literal
    day : Expr, str, int
        Column or literal

    Examples
    --------
    >>> df.mutate(date = tp.make_date(2000, 1, 1))
    """
    return pl.date(year, month, day)

def make_datetime(year = 1970, month = 1, day = 1, hour = 0, minute = 0, second = 0):
    """
    Create a datetime object

    Parameters
    ----------
    year : Expr, str, int
        Column or literal
    month : Expr, str, int
        Column or literal
    day : Expr, str, int
        Column or literal
    hour : Expr, str, int
        Column or literal
    minute : Expr, str, int
        Column or literal
    second : Expr, str, int
        Column or literal

    Examples
    --------
    >>> df.mutate(date = tp.make_datetime(2000, 1, 1))
    """
    return pl.datetime(year, month, day, hour, minute, second)

def minute(x):
    """
    Extract the minute from a datetime

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(hour = tp.minute(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.minute()

def month(x):
    """
    Extract the month from a date

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(year = tp.month(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.month()

def quarter(x):
    """
    Extract the quarter from a date

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(quarter = tp.quarter(tp.col('x')))
    """
    x = _col_expr(x)
    return (x.dt.month() // 4) + 1

_frequencies = {
    "year": "y",
    "quarter": "q",
    "month": "mo",
    "week": "w",
    "day": "d",
    "hour": "h",
    "minute": "m",
    "second": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns"
}

def dt_round(x, rule, n=1):
    """
    Round the datetime

    Parameters
    ----------
    x : Expr, Series
        Column to operate on
    rule : str
        Units of the downscaling operation.
        Any of:
            - "year"
            - "quarter"
            - "month"
            - "week"
            - "day"
            - "hour"
            - "minute"
            - "second"
            - "ms" (milisecond)
            - "us" (microsecond)
            - "ns" (nanosecond)
    n : int
        Number of units (e.g. 5 "day", 15 "minute".

    Examples
    --------
    >>> df.mutate(monthday = tp.mday(tp.col('x')))
    """
    x = _col_expr(x)
    freq = _frequencies[rule]
    every = str(n) + freq
    return x.dt.round(every)

def second(x):
    """
    Extract the second from a datetime

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(hour = tp.minute(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.second()

def wday(x):
    """
    Extract the weekday from a date from sunday = 1 to saturday = 7.

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(weekday = tp.wday(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.weekday() + 1

def week(x):
    """
    Extract the week from a date

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(week = tp.week(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.week()

def yday(x):
    """
    Extract the year day from a date from 1 to 366.

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(yearday = tp.yday(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.ordinal_day()

def year(x):
    """
    Extract the year from a date

    Parameters
    ----------
    x : Expr, Series
        Column to operate on

    Examples
    --------
    >>> df.mutate(year = tp.year(tp.col('x')))
    """
    x = _col_expr(x)
    return x.dt.year()
