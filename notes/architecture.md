# TABLE OF CONTENTS
- [1. reexports.py](#1-reexportspy)
- [2. utils.py](#2-utilspy)
- [3. tidyselect.py](#3-tidyselectpy)
- [4. stringr.py](#4-stringrpy)
- [5. lubridate.py](#5-lubridatepy)

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

## 2.3. _list_flatten(x)
Convert nested list into standard list.   
For example: [[1, 2, 3], [4], [5, 6]] -> [1, 2, 3, 4, 5, 6]

## 2.4. _as_list(x)
Convert given inputs into a list   
For example:
  + [[1, 2, 3], [4], [5, 6]] -> [1, 2, 3, 4, 5, 6]
  + pl.Int8 -> [pl.Int8]
  + 3 -> [3]
  + (3, 2, 5, 0) -> [3, 2, 5, 0]
  + None -> []

## 2.5. _repeat(x, times)
Repeat the input list a given times.   
If the input is not a list, then convert to list first.   

For example: ```_repeat(x=(1, 2), times=3)``` will return ```[1, 2, 1, 2, 1, 2]```

## 2.6. _str_to_lit(x):
if x is a string,    
then return ```pl.lit(x)```   
else return x   

For example: ```"active"``` -> ```pl.lit("active")```

## 2.7. _lit_expr(x):
if x is a single Python scalar value,   
then return ```pl.lit(x)```   
else return x   

For example: ```3``` -> ```pl.lit(3)```

## 2.8. _mutate_cols(df, exprs):
Mutate columns with given dataframe and expressions   
by using ```df.with_columns()``` method.

## 2.9. _col_expr(x):
Convert input x into ```pl.col(x)```

## 2.10. _col_exprs(x):
Convert elements of input list x into ```[pl.col(x[i])]```

## 2.11. _kwargs_as_exprs(kwargs):
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
