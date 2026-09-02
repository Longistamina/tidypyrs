# TABLE OF CONTENTS
- [1. reexports.py](#1-reexportspy)
- [2. utils.py](#2-utilspy)
- [3. tidyselect.py](#3-tidyselectpy)
- [4. stringr.py](#4-stringrpy)
- [5. lubridate.py](#5-lubridatepy)
- [6. groupby.py](#6-groupbypy)
- [7. tibble_frame.py](#7-tibbleframepy)
- [8. tibble_lazy.py](#8-tibblelazypy)
- [9. funs.py](#9-funspy)
- [10. f_namespace.py](#10-fnamespacepy)

# 1. reexports.py
Import essential Polars classes and functions that will become
part of the public ```tidypyrs``` library

# 2. utils.py
This is the most important file for understanding the backend.   
It is the normalization layer between Tidyverse syntax and Polars expressions.

## 2.1. _safe_len(x)
Return ```0``` if x is ```None```,   
otherwise return ```len(x)```

## 2.2. Type checking utilities
_is_boolean(x)   
_is_integer(x)   
_is_float(x)   
_is_string(x)   
_is_constant(x): True if x is boolean, integer, float or string   
_is_list(x)   
_is_tuple(x)   
_is_iterable(x): True if x has ```__iter__``` attribute but not a string   
_is_series(x): True if x is a ```pl.Series```
_is_type(x): True if x is Polars literal value like ```pl.Int8```, which has ```type(x).__name__ == 'DataTypeClass'```

## 2.3. _uses_over(over) and _over_exprs(exprs, over)
Check if user uses ```over``` parameter in any function that supports this parameter,
and convert given ```expr``` into grouped ```expr.over(over)```

## 2.4. _list_flatten(x)
Convert nested list into standard list.   
For example: [[1, 2, 3], [4], [5, 6]] -> [1, 2, 3, 4, 5, 6]

## 2.5. _as_list(x)
Convert given inputs into a list   
For example:
  + [[1, 2, 3], [4], [5, 6]] -> [1, 2, 3, 4, 5, 6]
  + pl.Int8 -> [pl.Int8]
  + 3 -> [3]
  + (3, 2, 5, 0) -> [3, 2, 5, 0]
  + None -> []

## 2.6. _repeat(x, times)
Repeat the input list a given times.   
If the input is not a list, then convert to list first.   

For example: ```_repeat(x=(1, 2), times=3)``` will return ```[1, 2, 1, 2, 1, 2]```

## 2.7. _str_to_lit(x):
if x is a string,    
then return ```pl.lit(x)```   
else return x   

For example: ```"active"``` -> ```pl.lit("active")```

## 2.8. _lit_expr(x):
if x is a single Python scalar value,   
then return ```pl.lit(x)```   
else return x   

For example: ```3``` -> ```pl.lit(3)```

## 2.9. _mutate_cols(df, exprs):
Mutate columns with given dataframe and expressions   
by using ```df.with_columns()``` method.

## 2.10. _col_expr(x):
Convert input x into ```pl.col(x)```

## 2.11. _col_exprs(x):
Convert elements of input list x into ```[pl.col(x[i])]```

## 2.12. _kwargs_as_exprs(kwargs):
Convert given key-value pairs into ```pl.lit(value).alias(key)```

# 3. tidyselect.py
This file maps tidyselect ideas to ```polars.selectors```   

It contains definitions of these APIs:
+ ```contains()```: contains a literal string
+ ```starts_with()```: starts with a prefix
+ ```ends_with()```: ends with a suffix
+ ```everything()```: selects all columns
+ ```where()```: select columns by type using a string like ```tp.where("string")```

# 4. stringr.py
This file maps Tidyverse string functions to ```polars`` APIs   

It contains these APIs:
+ ```str_length()```
+ ```str_to_lower()```
+ ```str_to_upper()```
+ ```str_paste()```
+ ```str_paste0()```
+ ```str_concat()```
+ ```str_detect()```
+ ```str_starts()```
+ ```str_ends()```
+ ```str_replace()```
+ ```str_replace_all()```
+ ```str_extract()```
+ ```str_sub()```
+ ```str_remove_all()```
+ ```str_remove()```
+ ```str_trim()```

# 5. lubridate.py
This file maps Tidyverse datetime functions to ```polars`` APIs   

It contains these APIs:
+ ```as_date()```
+ ```as_datetime()```
+ ```hour()```
+ ```make_date()```
+ ```make_datetime()```
+ ```mday()```
+ ```minute()```
+ ```month()```
+ ```quarter()```
+ ```dt_round()```
+ ```second()```
+ ```wday()```
+ ```week()```
+ ```yday()```
+ ```year()```

# 6. groupby.py
This file contains the definitions of ```TibbleGroupBy``` and ```TibbleLazyGroupBy```,   
mapping ```tidypyrs``` APIs to ```polars``` APIs.

# 7. tibble_frame.py
This file contains the definition of ```TibbleFrame``` and its tidyverse-style methods,   
mapping ```polars.DataFrame``` to ```TibbleFrame```

# 8. tibble_lazy.py
This file contains the definition of ```TibbleLazy``` and its tidyverse-style methods,   
mapping ```polars.LazyFrame``` to ```TibbleLazy```

# 9. funs.py
This file contains the definition of standalone functions   
that should be called directly from ```tidypyrs```   

For example:
```python
import tidypyrs as tp

tp.as_enum()
tp.as_categorical()
```

# 10. f_namespace.py
This file contains the definitions of ```_Deferred``` and ```_FrameReference``` classess   
which allow bypassing the need of ```.pipe()``` method in certain cases.   

Example:   
```python
import tidypyrs as tp
from tidypyrs import f

tl = (
    tp.TibbleLazy(y=["b", "a", "b"])
    .mutate(
        y=tp.as_ordered(f.select("y"))
        # Don't need to call `.pipe(lambda f: f.mutate(y = tp.as_ordered(f.select("y"))))`
    )
)
```
