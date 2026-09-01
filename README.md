# tidypyrs

tidypyrs is a data frame library built on top of the blazingly fast [polars](https://github.com/pola-rs/polars) library that gives access to methods and functions familiar to R tidyverse users.   

This library is inspired by [tidypolars](https://github.com/markfairbanks/tidypolars), as well as continues the work tidypolars left behind.

## Installation
You can install tidypyrs with `pip`:

```bash
$ pip install tidypyrs
```

Or through `uv`:
```bash
$ uv pip install tidypyrs
```

## Documentation
Detailed documentations will be added later in the near future.

## General syntax

tidypyrs methods are designed to work like tidyverse functions:

```python
import tidypyrs as tp
from tidypyrs import col as c

tf = tp.TibbleFrame(
    x = range(3), 
    y = range(3, 6), 
    z = ['a', 'a', 'b']
)

(
    tf
    .select('x', 'y', 'z')
    .filter(col('x') < 4, c('y') > 1)
    .arrange(desc('z'), 'x')
    .mutate(double_x = c('x') * 2,
            x_plus_y = c('x') + c('y'))
)
```

```
┌─────┬─────┬─────┬──────────┬──────────┐
│ x   ┆ y   ┆ z   ┆ double_x ┆ x_plus_y │
│ --- ┆ --- ┆ --- ┆ ---      ┆ ---      │
│ i64 ┆ i64 ┆ str ┆ i64      ┆ i64      │
╞═════╪═════╪═════╪══════════╪══════════╡
│ 2   ┆ 5   ┆ b   ┆ 4        ┆ 7        │
├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ 0   ┆ 3   ┆ a   ┆ 0        ┆ 3        │
├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ 1   ┆ 4   ┆ a   ┆ 2        ┆ 5        │
└─────┴─────┴─────┴──────────┴──────────┘
```

The key difference from R is that column names must be wrapped in `col()` in the following methods:
* `.filter()`
* `.mutate()`
* `.summarize()`

The general idea - when doing calculations on a column you need to wrap it in `col()`. When doing simple column selections (like in `.select()`) you can pass the column names as strings.

## Fast group-by syntax with ```over``` parameter

polars provides ```polars.Expr.over()``` to perform
quick and short group-by operation without calling ```group_by().agg()```.
Library tidypyrs supports the same idea with ```over``` parameter.

* A single column can be passed with `over = 'z'`
* Multiple columns can be passed with `over = ['y', 'z']`

```python
tf = tp.TibbleFrame({'x': range(3), 'y': ['a', 'a', 'b']})
print(
    tf
    .filter(c('x') <= c('x').mean(), over='y')
    .arrange('y')
)
```

```
┌─────┬─────┐
│ x   ┆ y   │
│ --- ┆ --- │
│ f64 ┆ str │
╞═════╪═════╡
│ 0   ┆ a   │
├╌╌╌╌╌┼╌╌╌╌╌┤
│ 2   ┆ b   │
└─────┴─────┘
```

## Selecting/dropping columns

tidyselect functions can be mixed with normal selection when selecting columns:

```python
tf = tp.TibbleFrame(x1 = range(3), x2 = range(3), y = range(3), z = range(3))

tf.select(tp.starts_with('x'), 'z')
```

```
┌─────┬─────┬─────┐
│ x1  ┆ x2  ┆ z   │
│ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ 0   ┆ 0   ┆ 0   │
├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
│ 1   ┆ 1   ┆ 1   │
├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
│ 2   ┆ 2   ┆ 2   │
└─────┴─────┴─────┘
```

To drop columns use the `.drop()` method:

```python
tf.drop(tp.starts_with('x'), 'z')
```

```
┌─────┐
│ y   │
│ --- │
│ i64 │
╞═════╡
│ 0   │
├╌╌╌╌╌┤
│ 1   │
├╌╌╌╌╌┤
│ 2   │
└─────┘
```

## Support TibbleLazy as a translated version of polars.LazyFrame
```python
tl = tp.TibbleLazy(
    group=["a", "a", "b"],
    value=[1, 2, 3],
)

result = tl.mutate(
    group_mean=tp.mean("value"),
    over="group",
)

print(result.collect())
```

```
┌───────┬───────┬────────────┐
│ group ┆ value ┆ group_mean │
│ ---   ┆ ---   ┆ ---        │
│ str   ┆ i64   ┆ f64        │
╞═══════╪═══════╪════════════╡
│ a     ┆ 1     ┆ 1.5        │
│ a     ┆ 2     ┆ 1.5        │
│ b     ┆ 3     ┆ 3.0        │
└───────┴───────┴────────────┘
```

Can call ```TibbleFrame.lazy()``` to convert it to TibbleLazy,
and can call ```TibbleLazy.collect()``` to realize back to TibbleFrame.

### "f" namespace for fast selecting and accessing columns
tidypyrs provides a very convenient "f" namespace to allow fast selecting and accessing columns.
This pushes ```polars.col()``` one step further from a mere column expression.
This feature is inspired by [datar](#https://github.com/pwwang/datar) library.
```python
import tidypyrs as tp
import polars as pl
from tidypyrs import f

tf = tp.TibbleFrame(
    y=["b", "a", "b"]
).mutate(y=tp.as_ordered(f.select("y"))) # No need to use .pipe(lambda f: f.mutate(y=tp.as_ordered(f.select("y"))))

print(isinstance(tf.pull("y").dtype, pl.Enum)) # True
```

## Converting to/from pandas data frames

If you need to use a package that requires pandas data frames, you can convert from a tidypyrs `TibbleFrame/TibbleLazy` to
a pandas `DataFrame`.

To do this you'll first need to install pyarrow:

```bash
pip install pyarrow
```

To convert to a pandas `DataFrame`:

```python
df = tf.as_pandas()
```

To convert from a pandas `DataFrame` to a tidypyrs `TibbleFrame/TibbleLazy`:

```python
tf = tp.as_tf(df) # TibbleFrame
tl = tp.as_tl(df) # TibbleLazy
```
