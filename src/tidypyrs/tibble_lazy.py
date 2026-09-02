import polars as pl
from datetime import timedelta
import functools as ft

from .utils import (
    _as_list,
    _col_exprs,
    _is_expr,
    _is_string,
    _kwargs_as_exprs,
    _mutate_cols,
    _uses_over,
    _over_exprs,
)
from .stringr import str_concat
from .groupby import TibbleLazyGroupBy
import copy
from .reexports import *
from .tidyselect import everything
from operator import not_, and_

__all__ = [
    "as_tl",
    "is_tl",
    "TibbleLazy",
]


class TibbleLazy(pl.LazyFrame):
    """
    A lazy data frame object that provides methods familiar to R tidyverse users.
    """

    def __init__(self, _data=None, **kwargs):
        if len(kwargs) > 0:
            _data = kwargs
        elif not_(isinstance(_data, dict)):
            raise ValueError("_data must be a dictionary or kwargs must be used")
        super().__init__(_data)

    def __dir__(self):
        _TibbleLazy_methods = [
            "arrange",
            "as_dict",
            "as_pandas",
            "as_polars",
            "bind_cols",
            "bind_rows",
            "clone",
            "collect",
            "colnames",
            "count",
            "distinct",
            "drop",
            "drop_null",
            "head",
            "fill",
            "filter",
            "glimpse",
            "group_by",
            "group_by_dynamic",
            "inner_join",
            "left_join",
            "mutate",
            "full_join",
            "pivot_longer",
            "pivot_wider",
            "print",
            "relocate",
            "rename",
            "replace_null",
            "select",
            "separate",
            "set_names",
            "slice",
            "slice_head",
            "slice_tail",
            "summarize",
            "tail",
        ]
        return _TibbleLazy_methods

    def __repr__(self):
        """Printing method"""
        tl = self.as_polars()
        return tl.__str__()

    def _repr_html_(self):
        """
        Printing method for jupyter

        Output rows and columns can be modified by setting the following ENVIRONMENT variables:

        * POLARS_FMT_MAX_COLS: set the number of columns

        * POLARS_FMT_MAX_ROWS: set the number of rows
        """
        tl = self.as_polars()
        return tl._repr_html_()

    def __copy__(self):
        # Shallow copy
        # See: https://stackoverflow.com/a/51043609/13254470
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __str__(self):
        """Printing method"""
        tl = self.as_polars()
        return tl.__str__()

    def __getattribute__(self, attr):
        if attr in _polars_methods:
            raise AttributeError
        return pl.LazyFrame.__getattribute__(self, attr)

    def arrange(self, *args):
        """
        Arrange/sort rows

        Parameters
        ----------
        *args : str
            Columns to sort by

        Examples
        --------
        >>> tl = tp.TibbleLazy({'x': ['a', 'a', 'b'], 'y': range(3)})
        >>> # Arrange in ascending order
        >>> tl.arrange('x', 'y')
        ...
        >>> # Arrange some columns descending
        >>> tl.arrange(tp.desc('x'), 'y')
        """
        from .funs import DescCol

        exprs = _as_list(args)
        desc = [bool(isinstance(expr, DescCol)) for expr in exprs]
        return super().sort(exprs, descending=desc).pipe(_from_polars_lazy)

    def as_polars(self):
        """
        Convert to a polars DataFrame

        Examples
        --------
        >>> tl.as_polars()
        """
        self = copy.copy(self)
        self.__class__ = pl.LazyFrame
        return self

    def bind_cols(self, *args):
        """
        Bind data frames by columns

        Parameters
        ----------
        tl : tibble
            Data frame to bind

        Examples
        --------
        >>> tl1 = tp.TibbleLazy({'x': ['a', 'a', 'b'], 'y': range(3)})
        >>> tl2 = tp.TibbleLazy({'a': ['c', 'c', 'c'], 'b': range(4, 7)})
        >>> tl1.bind_cols(tl2)
        """
        frames = _as_list(args)
        out = self.as_polars()
        for frame in frames:
            out = out.hstack(frame)
        return out.pipe(_from_polars_lazy)

    def bind_rows(self, *args):
        """
        Bind data frames by row

        Parameters
        ----------
        *args : tibble, list
            Data frames to bind by row

        Examples
        --------
        >>> tl1 = tp.TibbleLazy({'x': ['a', 'a', 'b'], 'y': range(3)})
        >>> tl2 = tp.TibbleLazy({'x': ['c', 'c', 'c'], 'y': range(4, 7)})
        >>> tl1.bind_rows(tl2)
        """
        frames = _as_list(args)
        out = pl.concat([self, *frames], how="diagonal")
        return out.pipe(_from_polars_lazy)

    def clone(self):
        """Very cheap deep clone"""
        return super().clone().pipe(_from_polars_lazy)

    def collect(self, engine="auto"):
        "Collect the TibbleLazy with selected engine and return TibbleFrame"
        from tidypyrs.tibble_frame import _from_polars_frame

        return super().collect(engine=engine).pipe(_from_polars_frame)

    def count(self, *args, sort=False, name="n"):
        """
        Returns row counts of the dataset.
        If bare column names are provided, count() returns counts by group.

        Parameters
        ----------
        *args : str, Expr
            Columns to group by
        sort : bool
            Should columns be ordered in descending order by count
        name : str
            The name of the new column in the output. If omitted, it will default to "n".

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': ['a', 'a', 'b']})
        >>> tl.count()
        >>> tl.count('b')
        """
        args = _as_list(args)

        out = self.group_by(args).summarise(pl.len()).alias(name)

        if sort == True:
            from .funs import desc

            out = out.arrange(desc(name))

        return out

    def distinct(self, *args):
        """
        Select distinct/unique rows

        Parameters
        ----------
        *args : str, Expr
            Columns to find distinct/unique rows

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': ['a', 'a', 'b']})
        >>> tl.distinct()
        >>> tl.distinct('b')
        """
        args = _as_list(args)
        if len(args) == 0:
            tl = super().unique()
        else:
            tl = super().select(args).unique()
        return tl.pipe(_from_polars_lazy)

    def drop(self, *args, strict=True):
        """
        Drop unwanted columns

        Parameters
        ----------
        *args : str
            Columns to drop

        Examples
        --------
        >>> tl.drop('x', 'y')
        """
        drop_cols = _as_list(args)
        return super().drop(drop_cols, strict=strict).pipe(_from_polars_lazy)

    def drop_null(self, *args):
        """
        Drop rows containing missing values

        Parameters
        ----------
        *args : str
            Columns to drop nulls from (defaults to all)

        Examples
        --------
        >>> tl = tp.TibbleLazy(x = [1, None, 3], y = [None, 'b', 'c'], z = range(3)}
        >>> tl.drop_null()
        >>> tl.drop_null('x', 'y')
        """
        args = _as_list(args)
        if len(args) == 0:
            out = super().drop_nulls()
        else:
            out = super().drop_nulls(args)
        return out.pipe(_from_polars_lazy)

    def equals(self, other, null_equal=True):
        """
        Check if two TibbleLazy are equal
        Note: this requires calling `collect()` to realize the TibbleFrame
        """
        tl = self.as_polars().collect()
        other = other.as_polars().collect()
        return tl.equals(other, null_equal=null_equal)

    def glimpse(self):
        """
        Return a dense preview of the DataFrame.

        The formatting shows one line per column so that wide dataframes display cleanly.
        Each line shows the column name, the data type, and the first few values.
        """
        return self.as_polars().glimpse()

    def group_by(
        self, *by, maintain_order: bool = False, **named_by
    ) -> TibbleLazyGroupBy:
        """
        Start a group by operation.

        Parameters
        ----------
        *by
            Column(s) to group by. Accepts expression input. Strings are parsed as
            column names.
        maintain_order
            Ensure that the order of the groups is consistent with the input data.
            This is slower than a default group by.
            Settings this to `True` blocks the possibility
            to run on the streaming engine.

            .. note::
                Within each group, the order of rows is always preserved, regardless
                of this argument.
        **named_by
            Additional columns to group by, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Returns
        -------
        TibbleLazyGroupBy
            Object which can be used to perform aggregations with ``agg``, ``summarise`` or ``summarize``.
        """

        group_by = self.as_polars().group_by(
            *by,
            maintain_order=maintain_order,
            **named_by,
        )
        return TibbleLazyGroupBy(group_by, _from_polars_lazy)

    def group_by_dynamic(
        self,
        index_column,
        *,
        every: str | timedelta,
        period: str | timedelta | None = None,
        offset: str | timedelta | None = None,
        include_boundaries: bool = False,
        closed="left",
        label="left",
        group_by=None,
        start_by="window",
    ) -> TibbleLazyGroupBy:
        """
        Group based on a time value (or index value of type Int32, Int64).

        Time windows are calculated and rows are assigned to windows. Different from a
        normal group by is that a row can be member of multiple groups.
        By default, the windows look like:

        - [start, start + period)
        - [start + every, start + every + period)
        - [start + 2*every, start + 2*every + period)
        - ...

        where `start` is determined by `start_by`, `offset`, `every`, and the earliest
        datapoint. See the `start_by` argument description for details.

        .. warning::
            The index column must be sorted in ascending order. If `group_by` is passed, then
            the index column must be sorted in ascending order within each group.

        .. versionchanged:: 0.20.14
            The `by` parameter was renamed `group_by`.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `group_by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a dynamic group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        every
            interval of the window
        period
            length of the window, if None it will equal 'every'
        offset
            offset of the window, does not take effect if `start_by` is 'datapoint'.
            Defaults to zero.
        include_boundaries
            Add the lower and upper bound of the window to the "_lower_boundary" and
            "_upper_boundary" columns. This will impact performance because it's harder to
            parallelize
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        label : {'left', 'right', 'datapoint'}
            Define which label to use for the window:

            - 'left': lower boundary of the window
            - 'right': upper boundary of the window
            - 'datapoint': the first value of the index column in the given window.
              If you don't need the label to be at one of the boundaries, choose this
              option for maximum performance
        group_by
            Also group by this column/these columns
        start_by : {'window', 'datapoint', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
            The strategy to determine the start of the first window by.

            * 'window': Start by taking the earliest timestamp, truncating it with
              `every`, and then adding `offset`.
              Note that weekly windows start on Monday.
            * 'datapoint': Start from the first encountered data point.
            * a day of the week (only takes effect if `every` contains `'w'`):

              * 'monday': Start the window on the Monday before the first data point.
              * 'tuesday': Start the window on the Tuesday before the first data point.
              * ...
              * 'sunday': Start the window on the Sunday before the first data point.

              The resulting window is then shifted back until the earliest datapoint
              is in or in front of it.

        Returns
        -------
        TibbleLazyGroupBy
            Object you can call `.agg` or ``.summarise`` or ``.summarize`` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `group_by` columns are
            passed, it will only be sorted within each group).
        """
        grouped = super().group_by_dynamic(
            index_column,
            every=every,
            period=period,
            offset=offset,
            include_boundaries=include_boundaries,
            closed=closed,
            label=label,
            group_by=group_by,
            start_by=start_by,
        )

        return TibbleLazyGroupBy(grouped, _from_polars_lazy)

    def fill(self, *args, direction="down", over=None):
        """
        Fill in missing values with previous or next value

        Parameters
        ----------
        *args : str
            Columns to fill
        direction : str
            Direction to fill. One of ['down', 'up', 'downup', 'updown']
        over : str, list
            Columns to group over

        Examples
        --------
        >>> tl = tp.TibbleLazy({
        ...     'a': [1, None, 3, 4, 5],
        ...     'b': [None, 2, None, None, 5],
        ...     'groups': ['a', 'a', 'a', 'b', 'b']
        ... })
        >>> tl.fill('a', 'b')
        >>> tl.fill('a', 'b', over='groups')
        >>> tl.fill('a', 'b', over='groups')
        >>> tl.fill('a', 'b', direction='downup')
        """
        args = _as_list(args)
        if len(args) == 0:
            return self
        args = _col_exprs(args)
        options = {"down": "forward", "up": "backward"}
        if direction in ["down", "up"]:
            direction = options[direction]
            exprs = [arg.fill_null(strategy=direction) for arg in args]
        elif direction == "downup":
            exprs = [
                arg.fill_null(strategy="forward").fill_null(strategy="backward")
                for arg in args
            ]
        elif direction == "updown":
            exprs = [
                arg.fill_null(strategy="backward").fill_null(strategy="forward")
                for arg in args
            ]
        else:
            raise ValueError("direction must be one of down, up, downup, or updown")

        return self.mutate(*exprs, over=over)

    def filter(self, *conditions, over=None):
        """
        Filter rows on one or more conditions

        Parameters
        ----------
        *conditions : Expr
            Conditions to filter by
        over : str, list
            Columns to group by

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': ['a', 'a', 'b']})
        >>> tl.filter(col('a') < 2, col('b') == 'a')
        >>> tl.filter((col('a') < 2) & (col('b') == 'a'))
        >>> tl.filter(col('a') <= tp.mean(col('a')), over='b')
        """
        predicate = ft.reduce(and_, conditions)

        if _uses_over(over):
            predicate = predicate.over(_as_list(over))

        out = super().filter(predicate)
        return out.pipe(_from_polars_lazy)

    def full_join(
        self, tl, left_on=None, right_on=None, on=None, suffix: str = "_right"
    ):
        """
        Perform an full join

        Parameters
        ----------
        tl : tibble
            Lazy DataFrame to join with.
        left_on : str, list
            Join column(s) of the left DataFrame.
        right_on : str, list
            Join column(s) of the right DataFrame.
        on: str, list
            Join column(s) of both DataFrames. If set, `left_on` and `right_on` should be None.
        suffix : str
            Suffix to append to columns with a duplicate name.

        Examples
        --------
        >>> tl1.full_join(tl2)
        >>> tl1.full_join(tl2, on = 'x')
        >>> tl1.full_join(tl2, left_on='left_x', right_on='x')
        """
        if (left_on is None) & (right_on is None) & (on is None):
            on = list(set(self.colnames) & set(tl.colnames))
        out = super().join(
            tl,
            on,
            "full",
            left_on=left_on,
            right_on=right_on,
            suffix=suffix,
            coalesce=True,
        )
        return out.pipe(_from_polars_lazy)

    def head(self, n=5, *, over=None):
        """Alias for `.slice_head()`"""
        return self.slice_head(n, over=over).pipe(_from_polars_lazy)

    def inner_join(self, tl, left_on=None, right_on=None, on=None, suffix="_right"):
        """
        Perform an inner join

        Parameters
        ----------
        tl : tibble
            Lazy DataFrame to join with.
        left_on : str, list
            Join column(s) of the left DataFrame.
        right_on : str, list
            Join column(s) of the right DataFrame.
        on: str, list
            Join column(s) of both DataFrames. If set, `left_on` and `right_on` should be None.
        suffix : str
            Suffix to append to columns with a duplicate name.

        Examples
        --------
        >>> tl1.inner_join(tl2)
        >>> tl1.inner_join(tl2, on='x')
        >>> tl1.inner_join(tl2, left_on='left_x', right_on='x')
        """
        if (left_on is None) & (right_on is None) & (on is None):
            on = list(set(self.colnames) & set(tl.colnames))
        return (
            super()
            .join(tl, on, "inner", left_on=left_on, right_on=right_on, suffix=suffix)
            .pipe(_from_polars_lazy)
        )

    def left_join(self, tl, left_on=None, right_on=None, on=None, suffix="_right"):
        """
        Perform a left join

        Parameters
        ----------
        tl : tibble
            Lazy DataFrame to join with.
        left_on : str, list
            Join column(s) of the left DataFrame.
        right_on : str, list
            Join column(s) of the right DataFrame.
        on: str, list
            Join column(s) of both DataFrames. If set, `left_on` and `right_on` should be None.
        suffix : str
            Suffix to append to columns with a duplicate name.

        Examples
        --------
        >>> tl1.left_join(tl2)
        >>> tl1.left_join(tl2, on='x')
        >>> tl1.left_join(tl2, left_on='left_x', right_on='x')
        """
        if (left_on is None) & (right_on is None) & (on is None):
            on = list(set(self.colnames) & set(tl.colnames))
        return (
            super()
            .join(tl, on, "left", left_on=left_on, right_on=right_on, suffix=suffix)
            .pipe(_from_polars_lazy)
        )

    def mutate(self, *args, over=None, **kwargs):
        """
        Add or modify columns

        Parameters
        ----------
        *args : Expr
            Column expressions to add or modify
        by : str, list
            Columns to group by
        **kwargs : Expr
            Column expressions to add or modify

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': range(3), c=['a', 'a', 'b']})
        >>> tl.mutate(
        ...     double_a = col('a') * 2,
        ...     a_plus_b = col('a') + col('b')
        ... )
        >>> tl.mutate(row_num = row_number(), over='c')
        """
        exprs = _as_list(args) + _kwargs_as_exprs(kwargs)
        exprs = _over_exprs(exprs, over)

        out = _mutate_cols(self.as_polars(), exprs)
        return out.pipe(_from_polars_lazy)

    def pivot_longer(self, cols=None, names_to="name", values_to="value"):
        """
        Pivot data from wide to long

        Parameters
        ----------
        cols : Expr
            List of the columns to pivot. Defaults to all columns.
        names_to : str
            Name of the new "names" column.
        values_to: str
            Name of the new "values" column

        Examples
        --------
        >>> tl = tp.TibbleLazy({'id': ['id1', 'id2'], 'a': [1, 2], 'b': [1, 2]})
        >>> tl.pivot_longer(cols=['a', 'b'])
        >>> tl.pivot_longer(cols=['a', 'b'], names_to='stuff', values_to='things')
        """
        if cols is None:
            cols = everything()
        tl_cols = pl.Series(self.colnames)
        value_vars = self.select(cols).colnames
        id_vars = tl_cols.filter(tl_cols.is_in(value_vars).not_()).to_list()
        out = super().unpivot(
            index=id_vars, on=value_vars, variable_name=names_to, value_name=values_to
        )
        return out.pipe(_from_polars_lazy)

    def pivot_wider(
        self,
        names_from="name",
        names_list=None,
        values_from="value",
        id_cols=None,
        values_fn="first",
        values_fill=None,
    ):
        """
        Pivot data from long to wide

        Parameters
        ----------
        names_from : str
            Column to get the new column names from.
        names_list: Sequence[Any] | Series
            Desired output names for output columns (should be provided when using TibbleLazy)
            If not provided, the internal inference can be expensive
        values_from : str
            Column to get the new column values from
        id_cols : str, list
            A set of columns that uniquely identifies each observation.
            Defaults to all columns in the data table except for the columns specified in
            `names_from` and `values_from`.
        values_fn : str
            Function for how multiple entries per group should be dealt with.
            Any of 'first', 'count', 'sum', 'max', 'min', 'mean', 'median', 'last'
        values_fill : str
            If values are missing/null, what value should be filled in.
            Can use: "backward", "forward", "mean", "min", "max", "zero", "one"

        Examples
        --------
        >>> tl = tp.TibbleLazy({'id': [1, 1], 'variable': ['a', 'b'], 'value': [1, 2]})
        >>> tl.pivot_wider(names_from='variable', values_from='value')
        """
        if id_cols is None:
            tl_cols = pl.Series(self.colnames)
            from_cols = pl.Series(self.select(names_from, values_from).colnames)
            id_cols = tl_cols.filter(tl_cols.is_in(from_cols).not_()).to_list()

        if names_list is None:
            names_list = self.select(names_from).collect().to_series().unique()

        no_id = len(id_cols) == 0

        if no_id:
            id_cols = "_id"
            self = self.mutate(_id=pl.lit(1))

        out = (
            super()
            .pivot(
                values=values_from,
                index=id_cols,
                on=names_from,
                on_columns=names_list,
                aggregate_function=values_fn,
            )
            .pipe(_from_polars_lazy)
        )

        if values_fill != None:
            new_cols = pl.Series(out.colnames)
            new_cols = new_cols.filter(~new_cols.is_in(id_cols))
            fill_exprs = [col(new_col).fill_null(values_fill) for new_col in new_cols]
            out = out.mutate(*fill_exprs)

        if no_id:
            out = out.drop("_id")

        return out

    def print(self):
        self.pipe(print)

    def pull(self, var=None):
        """
        Extract a column as a series

        NOTICE 1: for TibbleFrame, this requires calling `.collect()`,
                  what it runs under the hood is `tl.select(var).collect().to_series()`

        NOTICE 2: for TibbleFrame, if `var` is not provided,
                  it will run `collect_schema()` under the hood to get column names

        Parameters
        ----------
        var : str
            Name of the column to extract. Defaults to the last column.

        Examples
        --------
        >>> tf = tp.TibbleFrame({'a': range(3), 'b': range(3))
        >>> tf.pull('a')
        """
        if var is None:
            var = self.colnames[-1]
        return super().select(var).collect().to_series()

    def relocate(self, *args, _before=None, _after=None):
        """
        Move a column or columns to a new position

        Parameters
        ----------
        *args : str, Expr
            Columns to move

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': range(3), 'c': ['a', 'a', 'b']})
        >>> tl.relocate('a', before='c')
        >>> tl.relocate('b', after='c')
        """
        cols_all = pl.Series(self.colnames)
        locs_all = pl.Series(range(len(cols_all)))
        locs_dict = {k: v for k, v in zip(cols_all, locs_all)}
        locs_tl = pl.DataFrame(locs_dict, orient="row")

        cols_relocate = _as_list(args)
        locs_relocate = pl.Series(locs_tl.select(cols_relocate).row(0))

        if len(locs_relocate) == 0:
            return self

        uses_before = _is_expr(_before) | _is_string(_before)
        uses_after = _is_expr(_after) | _is_string(_after)

        if uses_before & uses_after:
            raise ValueError("Cannot provide both before and after")
        elif not_(uses_before) & not_(uses_after):
            _before = cols_all[0]
            uses_before = True

        if uses_before:
            _before = locs_tl.select(_before).get_column(_before)
            locs_start = locs_all.filter(locs_all < _before)
        else:
            _after = locs_tl.select(_after).get_column(_after)
            locs_start = locs_all.filter(locs_all <= _after)

        locs_start = locs_start.filter(~locs_start.is_in(locs_relocate))
        final_order = pl.concat([locs_start, locs_relocate, locs_all]).unique(
            maintain_order=True
        )
        final_order = cols_all[final_order].to_list()

        return self.select(final_order)

    def rename(self, mapping=None, **kwargs):
        """
        Rename columns

        Parameters
        ----------
        mapping : dict
            Dictionary mapping of new names or a Callable function like lambda
        **kwargs : str
            key-value pair of new name from old name

        Examples
        --------
        >>> tl = tp.TibbleLazy({'x': range(3), 't': range(3), 'z': ['a', 'a', 'b']})
        >>> tl.rename(new_x = 'x') # dplyr interface
        >>> tl.rename({'x': 'new_x'}) # pandas interface
        """
        if mapping is None:
            mapping = {value: key for key, value in kwargs.items()}
        return super().rename(mapping).pipe(_from_polars_lazy)

    def replace_null(self, replace=None):
        """
        Replace null values

        Parameters
        ----------
        replace : dict
            Dictionary of column/replacement pairs

        Examples
        --------
        >>> tl = tp.TibbleLazy(x = [0, None], y = [None, None])
        >>> tl.replace_null(dict(x = 1, y = 2))
        """
        if replace is None:
            return self
        if type(replace) != dict:
            raise ValueError("replace must be a dictionary of column/replacement pairs")
        replace_exprs = [col(key).fill_null(value) for key, value in replace.items()]
        return self.mutate(*replace_exprs)

    def separate(self, sep_col, into, sep="_", remove=True):
        """
        Separate a character column into multiple columns

        Parameters
        ----------
        sep_col : str
            Column to split into multiple columns
        into : list
            List of new column names
        sep : str
            Separator to split on. Default to '_'
        remove : bool
            If True removes the input column from the output data frame

        Examples
        --------
        >>> tl = tp.TibbleLazy(x = ['a_a', 'b_b', 'c_c'])
        >>> tl.separate('x', into=['left', 'right'])
        """
        into_len = len(into) - 1
        sep_tl = (
            self.as_polars()
            .select(
                col(sep_col)
                .str.split_exact(sep, into_len)
                .alias("_seps")
                .struct.rename_fields(into)
            )
            .unnest("_seps")
            .pipe(_from_polars_lazy)
        )
        out = self.bind_cols(sep_tl)
        if remove == True:
            out = out.drop(sep_col)
        return out

    def set_names(self, nm=None):
        """
        Change the column names of the data frame

        Parameters
        ----------
        nm : list
            A list of new names for the data frame

        Examples
        --------
        >>> tl = tp.TibbleLazy(x = range(3), y = range(3))
        >>> tl.set_names(['a', 'b'])
        """
        if nm is None:
            nm = self.colnames
        nm = _as_list(nm)
        rename_dict = {k: v for k, v in zip(self.colnames, nm)}
        return self.rename(rename_dict)

    def select(self, *args):
        """
        Select or drop columns

        Parameters
        ----------
        *args : str, Expr
            Columns to select

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': range(3), 'c': ['a', 'a', 'b']})
        >>> tl.select('a', 'b')
        >>> tl.select(col('a'), col('b'))
        """
        args = _as_list(args)
        args = _col_exprs(args)
        return super().select(args).pipe(_from_polars_lazy)

    def slice(self, *args, over=None):
        """
        Grab rows from a data frame

        Parameters
        ----------
        *args : int, list
            Rows to grab
        by : str, list
            Columns to group by

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': range(3), 'c': ['a', 'a', 'b']})
        >>> tl.slice(0, 1)
        >>> tl.slice(0, over='c')
        """
        rows = _as_list(args)

        if _uses_over(over):
            tl = super().select(
                pl.all().gather(rows).over(over, mapping_strategy="explode")
            )
        else:
            tl = super().select(pl.all().gather(rows))
        return tl.pipe(_from_polars_lazy)

    def slice_head(self, n=5, *, over=None):
        """
        Grab top rows from a data frame

        Parameters
        ----------
        n : int
            Number of rows to grab
        over : str, list
            Columns to group by

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': range(3), 'c': ['a', 'a', 'b']})
        >>> tl.slice_head(2)
        >>> tl.slice_head(1, over='c')
        """
        if _uses_over(over):
            tl = super().select(pl.all().head(n).over(over, mapping_strategy="explode"))
        else:
            tl = super().head(n)
        return tl.pipe(_from_polars_lazy)

    def slice_tail(self, n=5, *, over=None):
        """
        Grab bottom rows from a data frame

        Parameters
        ----------
        n : int
            Number of rows to grab
        by : str, list
            Columns to group by

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': range(3), 'c': ['a', 'a', 'b']})
        >>> tl.slice_tail(2)
        >>> tl.slice_tail(1, over='c')
        """
        if _uses_over(over):
            tl = super().select(pl.all().tail(n).over(over, mapping_strategy="explode"))
        else:
            tl = super().tail(n)
        return tl.pipe(_from_polars_lazy)

    def summarise(self, *args, **kwargs):
        """Alias for `.summarize()`"""
        return self.summarize(*args, **kwargs)

    def summarize(self, *args, **kwargs):
        """
        Aggregate data with summary statistics
        (not like ``group_by`` or ``group_by_dynamic``)

        Parameters
        ----------
        *args : Expr
            Column expressions to add or modify

        **kwargs : Expr
            Column expressions to add or modify

        Examples
        --------
        >>> tl = tp.TibbleLazy({'a': range(3), 'b': range(3), 'c': ['a', 'a', 'b']})
        >>> tl.summarize(avg_a = tp.mean(col('a')))
        >>> tl.summarize(
        ...     avg_a = tp.mean(col('a')),
        ...     max_b = tp.max(col('b'))
        ... )
        """
        exprs = _as_list(args) + _kwargs_as_exprs(kwargs)
        out = super().select(exprs)
        return out.pipe(_from_polars_lazy)

    def tail(self, n=5, *, over=None):
        """Alias for `.slice_tail()`"""
        return self.slice_tail(n, over=over).pipe(_from_polars_lazy)

    def unite(self, col="_united", unite_cols=None, sep="_", remove=True):
        """
        Unite multiple columns by pasting strings together

        Parameters
        ----------
        col : str
            Name of the new column
        unite_cols : list
            List of columns to unite
        sep : str
            Separator to use between values
        remove : bool
            If True removes input columns from the data frame

        Examples
        --------
        >>> tl = tp.TibbleLazy(a = ["a", "a", "a"], b = ["b", "b", "b"], c = range(3))
        >>> tl.unite("united_col", unite_cols = ["a", "b"])
        """
        if unite_cols is None:
            unite_cols = self.colnames
        else:
            unite_cols = self.select(unite_cols).colnames
        _before = unite_cols[0]
        unite_cols = _col_exprs(unite_cols)
        out = self.mutate(str_concat(*unite_cols, sep=sep).alias(col))
        out = out.relocate(col, _before=_before)
        if remove == True:
            out = out.drop(unite_cols)
        return out

    @property
    def colnames(self):
        """
        Use `collect_schema()` to resolve for column names,
        return as pl.Series

        Examples
        --------
        >>> tl.colnames
        """
        return pl.Series(super().collect_schema().keys(), dtype=pl.String)

    @property
    def columns(self):
        """
        Use `collect_schema()` to resolve for column names,
        return as pl.Series

        Examples
        --------
        >>> tl.columns
        """
        return pl.Series(super().collect_schema().keys(), dtype=pl.String)


##--------------------------------------------------------------------------------------##


def as_tl(x):
    """
    Convert an object to a TibbleLazy

    Parameters
    ----------
    x : [pl.DataFrame, pd.DataFrame, dict]
        Object to convert to a tibble

    Examples
    --------
    >>> tp.as_tibble(polars_tl)
    """
    if isinstance(x, pl.LazyFrame):
        out = _from_polars_lazy(x)
    elif isinstance(x, dict):
        out = TibbleLazy(x)
    elif is_tl(x):
        out = x
    else:
        out = _from_polars_lazy(pl.from_dataframe(x).lazy())
    return out


def is_tl(x):
    """
    Is an object to a TibbleLazy

    Parameters
    ----------
    x : object

    Examples
    --------
    >>> tp.is_tibble(x)
    """
    return isinstance(x, TibbleLazy)


def _from_polars_lazy(lf):
    tl = copy.copy(lf)
    tl.__class__ = TibbleLazy
    return tl


_allowed_methods = ["dtypes", "frame_equal", "get_columns", "lazy", "pipe"]

_polars_methods = [
    "apply",
    "columns",
    "describe",
    "downsample",
    "drop_duplicates",
    "explode",
    "fill_nan",
    "fill_null",
    "find_idx_by_name",
    "fold",
    "get_column",
    "groupby",
    "hash_rows",
    "height",
    "hstack",
    "insert_at_idx",
    "interpolate",
    "is_duplicated",
    "is_unique",
    "join",
    "limit",
    "max",
    "mean",
    "median",
    "min",
    "n_chunks",
    "null_count",
    "quantile",
    "rechunk",
    "replace",
    "replace_at_idx",
    "row",
    "rows",
    "sample",
    "select_at_idx",
    "shape",
    "shift",
    "shift_and_fill",
    "shrink_to_fit",
    "std",
    "sum",
    # 'to_arrow',
    # 'to_dict',
    "to_dicts",
    "to_dummies",
    "to_ipc",
    "to_json",
    "to_numpy",
    "to_pandas",
    "to_parquet",
    "transpose",
    "unnest",
    "unpivot",
    "var",
    "width",
    "with_column",
    "with_columns",
    "with_column_renamed",
    "with_columns",
]
