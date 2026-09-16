# 0.9.1
+ Enhance `_as_list` to handle `range(start, stop, step)` as input

# 0.9.0
+ Add `tp.Config` as alias for `pl.Config`

# 0.8.0
+ Add `tp.nth()` and `f.nth()` for selecting columns with slice of indices
+ Add `TibbleFrame.describe()` and `TibbleLazy.describe()`
+ Add `TibbleFrame.null_count()` and `TibbleLazy.null_count()`
+ Add `fill_null` and `fill_nan` to both `TibbleFrame` and `TibbleLazy`
+ Add `transpose` to `TibbleFrame`

# 0.7.0
+ Add `f.colnames` for `f` namespace
+ Add `__getitem__` for `_Deferred`
+ Make `_Deferred` able to convert any method into a `_Deferred` object
+ Add `_select_cols` for `select` to be able to handle `_Deferred` inputs

# 0.6.6
+ Enable `mutate` to run in both modes `parallel=True` and `parallel=False` 

# 0.6.5
+ Import `Callable` and `Mapping` in TibbleLazy

# 0.6.4
+ Enhance `rename` method behaviour with `strict` parameter

# 0.6.3
+ Convert `tp.Utf8` into `tp.String`

# 0.6.2
+ Make `glimpse` method accept positional argument

# 0.6.1
+ Add parameters to `glimpse` method

# 0.6.0
+ Add `row_index` method for both TibbleFrame and TibbleLazy as a wrapper of polars `with_row_index`

# 0.5.3
+ Fix bugs of `slice_head` and `slice_tail`'s behaviours
+ Enhance `tp.as_enum` and `tp.as_ordered` so that they can be used like `series = tp.as_enum(series)`

# 0.5.2
+ Make `as_enum` and `as_ordered` handle duplicates `categories`

# 0.5.1
+ Make `as_enum` and `as_ordered` handle `categories` better when user explicitly provide `categories`

# 0.5.0
+ Enhance the behaviour of `_defer_aware` decorator
+ Add `f.pull(var)` method for `f` namespace

# 0.4.1
+ Fix bugs for `read_excel` and change its signatures
+ Change signatures of `read_csv`, `read_parquet` and `scan_csv`

# 0.4.0
+ Add `read_excel` function to `funs.py`
+ Add `scan_csv` function to `funs.py`

# 0.3.0
+ Add `TibbleFrame.pipe()` method in `tibble_frame.py`
+ Add `TibbleLazy.pipe()` method in `tibble_lazy.py`

# 0.2.0
+ Add `pl.all`, `pl.when`, `pl.struct`, `pl.element`, `pl.concat_list` to `reexports.py`
+ Add `f.all()` method as alias for `pl.all()`
+ Reorganize imports

# 0.1.1
+ Improve `f` namespace behaviour.   
+ Enable `f["a", "b"]`, `f("a", "b")`.   
+ Add alias `f.sl("a", "b")` to work like `f.select("a", "b")`.   
+ Add `notes/f_namespace_explain.md`
+ Modify formats to respect ruff and pyright.   
+ Add `notes/ChangeLog.md`   

# 0.1.0
+ Publish the package.
