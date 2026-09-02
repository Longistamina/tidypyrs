"""
uv run pytest tests/lazy/test_lubridate.py
"""

import tidypyrs as tp
from tidypyrs import col as c


def test_date():
    """Can do date operations"""
    tl = tp.TibbleLazy(x=["2021-01-01", "2021-10-01"]).mutate(
        date=c("x").str.strptime(tp.Date)
    )
    actual = tl.mutate(
        date_check=tp.as_date("x"),
        mday=tp.mday("date"),
        quarter=tp.quarter("date"),
        wday=tp.wday("date"),
        week=tp.week("date"),
        yday=tp.yday("date"),
        year=tp.year("date"),
    )
    expected = (
        tp.TibbleLazy(x=["2021-01-01", "2021-10-01"])
        .mutate(date=c("x").str.strptime(tp.Date))
        .mutate(
            date_check=c("date"),
            mday=c("date").dt.day(),
            quarter=tp.Series([1, 3]),
            wday=c("date").dt.weekday() + 1,
            week=c("date").dt.week(),
            yday=c("date").dt.ordinal_day(),
            year=c("date").dt.year(),
        )
    )
    assert actual.equals(expected), "date operations failed"


def test_as_date_format():
    """Can pass fmt to as_date"""
    tl = tp.TibbleLazy(date=["12/31/2021"])
    out = tl.mutate(date_parsed=tp.as_date(c("date"), format="%m/%d/%Y"))
    assert out.pull().dtype == tp.Date, "as_date format failed"


def test_make_date():
    tl = tp.TibbleLazy(date=["2021-12-1"]).mutate(date=tp.as_date("date"))
    out = tl.mutate(date=tp.make_date(2021, 12, 1))
    assert tl.pull("date").equals(out.pull("date")), "make_date failed"
