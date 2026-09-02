"""
uv run pytest tests/frame/test_funs.py
"""

import tidypyrs as tp
from tidypyrs import col as c
import polars.selectors as cs
import math


def test_abs():
    """Can get absolute value"""
    tf = tp.TibbleFrame(x=range(-3, 0))
    actual = tf.mutate(abs_x=tp.abs("x"), abs_col_x=tp.abs(c("x")))
    expected = tp.TibbleFrame(
        x=range(-3, 0), abs_x=range(3, 0, -1), abs_col_x=range(3, 0, -1)
    )
    assert actual.equals(expected), "abs failed"


def test_agg_stats():
    """Can get aggregation statistics"""
    tf = tp.TibbleFrame(x=range(3), y=[2, 1, 0])
    actual = tf.summarize(
        corr=tp.cor("x", "y"),
        count_x=tp.count("x"),
        count_col_x=tp.count(c("x")),
        cov=tp.cov("x", "y"),
        first_x=tp.first("x"),
        first_col_x=tp.first(c("x")),
        last_x=tp.last("x"),
        last_col_x=tp.last(c("x")),
        max_x=tp.max("x"),
        max_col_x=tp.max(c("x")),
        mean_x=tp.mean("x"),
        mean_col_x=tp.mean(c("x")),
        median_x=tp.median("x"),
        median_col_x=tp.median(c("x")),
        min_x=tp.min("x"),
        min_col_x=tp.min(c("x")),
        n=tp.n(),
        n_distinct_x=tp.n_distinct("x"),
        n_distinct_col_x=tp.n_distinct(c("x")),
        quantile_x=tp.quantile("x", 0.25),
        sd_x=tp.sd("x"),
        sd_col_x=tp.sd(c("x")),
        sum_x=tp.sum("x"),
        sum_col_x=tp.sum(c("x")),
        var_y=tp.var("y"),
    ).mutate(tp.as_integer(cs.numeric().as_expr()))
    expected = tp.TibbleFrame(
        corr=[-1],
        count_x=[3],
        count_col_x=[3],
        cov=[-1],
        first_x=[0],
        first_col_x=[0],
        last_x=[2],
        last_col_x=[2],
        max_x=[2],
        max_col_x=[2],
        mean_x=[1],
        mean_col_x=[1],
        median_x=[1],
        median_col_x=[1],
        min_x=[0],
        min_col_x=[0],
        n=[3],
        n_distinct_x=[3],
        n_distinct_col_x=[3],
        quantile_x=[1],
        sd_x=[1],
        sd_col_x=[1],
        sum_x=[3],
        sum_col_x=[3],
        var_y=[1],
    )
    assert actual.equals(expected), "aggregation stats failed"


def test_as_factor():
    """Can use as_factor"""
    tf = tp.TibbleFrame(x=range(0, 10, 2), y=["a", "b", "b", "c", "a"]).mutate(
        y=tp.as_factor(c("y"))
    )

    assert tf.pull("y").dtype == tp.Categorical, "as_factor failed"


def test_as_ordered():
    """Can use as_ordered"""
    tf = tp.TibbleFrame(x=range(0, 10, 2), y=["a", "b", "b", "c", "a"])
    tf = tf.mutate(y=tp.as_ordered(tf.select("y")))
    assert tf.pull("y").dtype == tp.Enum, "as_ordered failed"


def test_as_ordered_reverse():
    """Can use as_ordered(reverse=True)"""
    tf = tp.TibbleFrame(x=range(0, 10, 2), y=["a", "b", "b", "c", "a"])
    tf = tf.mutate(y=tp.as_ordered(tf.select("y"), reverse=True))

    actual = tf.pull("y").dtype.categories
    expected = tp.Series(["c", "b", "a"])
    assert actual.equals(expected), "as_ordered(reverse=True) failed"


def test_case_when():
    """Can use case_when"""
    tf = tp.TibbleFrame(x=range(1, 4))
    actual = tf.mutate(case_x=tp.case_when(c("x") < 2, 0, c("x") < 3, 1, _default=0))
    expected = tp.TibbleFrame(x=range(1, 4), case_x=[0, 1, 0])
    assert actual.equals(expected), "case_when failed"


def test_casting():
    """Can do type casting"""
    tf = tp.TibbleFrame(
        int_col=[0, 0, 1], float_col=[1.0, 2.0, 3.0], chr_col=["1", "2", "3"]
    )
    actual = tf.mutate(
        float_cast=tp.as_float("int_col"),
        int_cast=tp.as_integer("float_col"),
        string_cast=tp.as_string("int_col"),
        bool_cast=tp.as_boolean("int_col"),
    ).select("float_cast", "int_cast", "string_cast", "bool_cast")
    expected = tp.TibbleFrame(
        float_cast=[0.0, 0.0, 1.0],
        int_cast=[1, 2, 3],
        string_cast=["0", "0", "1"],
        bool_cast=[False, False, True],
    )
    assert actual.equals(expected), "casting failed"


def test_coalesce():
    """Can use coalesce"""
    tf = tp.TibbleFrame(x=[None, None, 1], y=[2, None, 2], z=[3, 3, 3])
    actual = tf.mutate(coalesce_x=tp.coalesce(c("x"), c("y"), c("z"))).select(
        "coalesce_x"
    )
    expected = tp.TibbleFrame(coalesce_x=[2, 3, 1])
    assert actual.equals(expected), "coalesce failed"


def test_floor():
    """Can get the floor"""
    tf = tp.TibbleFrame(x=[1.1, 5.5])
    actual = tf.mutate(floor_x=tp.floor("x")).select("floor_x")
    expected = tp.TibbleFrame(floor_x=[1.0, 5.0])
    assert actual.equals(expected), "floor failed"


def test_lag():
    """Can get lagging values with function"""
    tf = tp.TibbleFrame({"x": range(3)})
    actual = tf.mutate(lag_null=tp.lag(c("x")), lag_default=tp.lag("x", default=1))
    expected = tp.TibbleFrame(
        {"x": range(3), "lag_null": [None, 0, 1], "lag_default": [1, 0, 1]}
    )
    assert actual.equals(expected, null_equal=True), "lag failed"


def test_lead():
    """Can get leading values with function"""
    tf = tp.TibbleFrame({"x": range(3)})
    actual = tf.mutate(lead_null=tp.lead(c("x")), lead_default=tp.lead("x", default=1))
    expected = tp.TibbleFrame(
        {"x": range(3), "lead_null": [1, 2, None], "lead_default": [1, 2, 1]}
    )
    assert actual.equals(expected, null_equal=True), "lead failed"


def test_logs():
    """Can get leading values with function"""
    tf = tp.TibbleFrame({"x": range(1, 4)})
    actual = tf.mutate(log=tp.log(c("x")).round(2), log10=tp.log10("x").round(2))
    expected = tf.mutate(log=c("x").log().round(2), log10=c("x").log10().round(2))
    assert actual.equals(expected), "log failed"


def test_if_else():
    """Can use if_else"""
    tf = tp.TibbleFrame(x=range(1, 4))
    actual = tf.mutate(case_x=tp.if_else(c("x") < 2, 1, 0))
    expected = tp.TibbleFrame(x=range(1, 4), case_x=[1, 0, 0])
    assert actual.equals(expected), "if_else failed"


def test_is_predicates():
    """Can use is predicates"""
    tf = tp.TibbleFrame(x=[0.0, 1.0, 2.0], y=[None, math.inf, math.nan])
    actual = (
        tf.mutate(
            between=tp.between("x", 1, 2),
            is_finite=tp.is_finite("x"),
            is_in=tp.is_in("x", [1.0, 2.0]),
            is_infinite=tp.is_infinite("y"),
            is_not=tp.is_not(tp.is_finite(c("x"))),
            is_not_in=tp.is_not_in("x", [1.0, 2.0]),
            is_not_null=tp.is_not_null("y"),
            is_null=tp.is_null("y"),
        )
    ).drop("x", "y")
    expected = tp.TibbleFrame(
        between=[False, True, True],
        is_finite=[True, True, True],
        is_in=[False, True, True],
        is_infinite=[None, True, False],
        is_not=[False, False, False],
        is_not_in=[True, False, False],
        is_not_null=[False, True, True],
        is_null=[True, False, False],
    )
    assert actual.equals(expected, null_equal=True), "is_predicates failed"


def test_rep():
    tf = tp.TibbleFrame(x=[0, 1], y=[0, 1])
    assert tp.rep(tf, 2).equals(tf.bind_rows(tf)), "rep tf failed"
    assert tp.rep(1, 2).equals(tp.Series([1, 1])), "rep int failed"
    assert tp.rep("a", 2).equals(tp.Series(["a", "a"])), "rep str failed"
    assert tp.rep(True, 2).equals(tp.Series([True, True])), "rep bool failed"
    assert tp.rep(tp.Series([0, 1]), 2).equals(tp.Series([0, 1, 0, 1])), (
        "rep series failed"
    )


def test_replace_null():
    """Can replace nulls"""
    tf = tp.TibbleFrame(x=[0, None], y=[None, None])
    actual = tf.mutate(x=tp.replace_null(c("x"), 1))
    expected = tp.TibbleFrame(x=[0, 1], y=[None, None])
    assert actual.equals(expected), "replace_null function failed"


def test_row_number():
    """Can get row number"""
    tf = tp.TibbleFrame(x=["a", "a", "b"])
    actual = tf.mutate(row_num=tp.row_number())
    expected = tp.TibbleFrame(x=["a", "a", "b"], row_num=[1, 2, 3])
    assert actual.equals(expected), "row_number failed"


def test_row_number_group():
    """Can get row number by group"""
    tf = tp.TibbleFrame(x=["a", "a", "b"])
    actual = tf.mutate(group_row_num=tp.row_number(), over="x").arrange(
        "x", "group_row_num"
    )
    expected = tp.TibbleFrame(x=["a", "a", "b"], group_row_num=[1, 2, 1])
    assert actual.equals(expected), "group row_number failed"


def test_round():
    """Can round values"""
    tf = tp.TibbleFrame(x=[1.11, 2.22, 3.33])
    actual = tf.mutate(x=tp.round(c("x"), 1))
    expected = tp.TibbleFrame(x=[1.1, 2.2, 3.3])
    assert actual.equals(expected), "round failed"


def test_sqrt():
    """Can get the square root"""
    tf = tp.TibbleFrame(x=[9, 25, 100])
    actual = tf.mutate(x=tp.sqrt("x"))
    expected = tp.TibbleFrame(x=[3, 5, 10])
    assert actual.equals(expected), "sqrt failed"
