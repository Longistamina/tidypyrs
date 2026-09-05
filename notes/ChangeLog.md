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
