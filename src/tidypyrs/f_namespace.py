import polars as pl
from collections.abc import Callable
from typing import Any

# ======================================
# define _Deferred class
# ======================================


class _Deferred:
    """An operation waiting for a DataFrame or LazyFrame."""

    __slots__ = ("_resolver",)

    def __init__(self, resolver: Callable[[Any], Any]):  # `resolver` function takes one argument of any type [Any], and return one value of any type Any
        self._resolver = resolver

    def resolve(self, frame):
        return self._resolver(frame)

    def map(self, function):
        return _Deferred(lambda frame: function(self.resolve(frame)))

    def alias(self, name):
        return self.map(lambda expression: expression.alias(name))


# =============================================
# define _FrameReference for ``f`` namespace
# =============================================


class _FrameReference:
    __slots__ = ()

    def __getitem__(self, *names) -> pl.Expr:
        """
        Return a column expression.

        -----------------------------
        Examples:
            f["a"] -> pl.col("a")
            f["a", "b"] -> pl.col(["a", "b"])
        """
        return pl.col(*names)

    def __getattr__(self, name) -> pl.Expr:
        """
        Allow f.column_name syntax.

        --------------------------------
        Example:
            f.a -> pl.col("a")
        """
        if name.startswith("_"):
            raise AttributeError(name)

        return pl.col(name)

    def __call__(self, *names) -> pl.Expr:
        """
        Allow f("col") syntax.

        --------------------------------
        Examples:
            f("a") -> pl.col("a")
            f("a", "b", "c") -> pl.col("a", "b", "c")
        """
        return pl.col(*names)

    def select(self, *exprs, **named_exprs) -> _Deferred:
        """
        Select from the frame currently executing the verb.
        Help eliminate pipe functions

        ------------------------------------------------------
        Example:
            tl = tp.TibbleLazy(
                x = range(0, 10, 2),
                y = ["a", "b", "b", "c", "a"]
            ).mutate(
                y = tp.as_enum(f.select("y"))
            )
        """
        return _Deferred(lambda frame: frame.select(*exprs, **named_exprs))

    sl = select  # Allows `f.sl("a")`

    def __repr__(self):
        return "f"


f = _FrameReference()

__all__ = ["f"]

"""
# =========================================================================
# EXPLANATION OF F_NAMESPACE
# =========================================================================

##--------------------------##
## 1. The goal of this file ##
##--------------------------##

Most Tidypyrs functions work directly with Polars expressions:
```
tf.mutate(doubled=tp.col("x") * 2)
```

However, functions such as ``tp.as_enum()`` and ``tp.as_ordered()`` can only infer
their categories from a single-column DataFrame or LazyFrame, not merely from a column expression
(this happens when the list of categories are not given):
```
tp.as_enum(tf.select("y"))
```

This is inconvenient inside a method chain because the current frame must be
named before we can select from it, so we might have to write:
```
tp.TibbleFrame(
    y=["b", "a", "b"]
).pipe(
    lambda tf: tf.mutate(
        copied=tp.col("y"),
        ordered=tp.as_enum(tf.select("copied")),
    )
)
```

The purpose of ``f_namespace.py`` is to make the same operation possible
without ``.pipe(lambda tf: ...)``:
```
tp.TibbleFrame(
    y=["b", "a", "b"]
).mutate(
    copied=tp.col("y"),
    ordered=tp.as_enum(f.select("copied")),
)
```

The ``f`` namespace provides two related features:

1. concise column expressions such as ``f["x"]`` and ``f.x``;
2. deferred frame operations such as ``f.select("x")``.

The second feature is why ``_Deferred`` is necessary.

##-----------------------------------------##
## 2. Why the current frame is unavailable ##
##-----------------------------------------##

Consider:
```
tf.mutate(
    y=tp.as_enum(f.select("y"))
)
```

Before Python can execute the body of ``tf.mutate()``, it must evaluate the
arguments passed to the method. The relevant flow is:
```
f.select("y")
    -> tp.as_enum(...)
        -> tf.mutate(y=...)
```

Python needs the result of ``f.select("y")`` before it can call ``tp.as_enum()``,
and it needs the result of ``tp.as_enum()`` before it can call the body of ``tf.mutate()``.

Therefore, when ``f.select("y")`` runs, ``mutate()`` has not yet executed,
so it cannot supply its current DataFrame or LazyFrame to ``f``.

So, ``f`` is left unknown and undefined, and ```f.select("y")```
becomes impossible to run.

Hence, we first need a real, defined ``f`` object,
and then a way for it to describe a selection whose frame will only become known later.

##-----------------------------------------------------------##
## 3. ``_FrameReference`` defines the public ``f`` namespace ##
##-----------------------------------------------------------##

The first problem is that ``f`` must be an actual Python object with known behavior.
This is solved by ``_FrameReference``:
```
class _FrameReference:
    __slots__ = ()

    ...

f = _FrameReference()
```

The class defines what operations such as ``f["x"]``, ``f.x``, and ``f.select("x")`` mean.
The final line creates one public instance named ``f``.

After this assignment, ``f`` is no longer unknown or undefined.
Python knows that it is an instance of ``_FrameReference``
and can look up the behavior defined by that class.

However, ``f`` does not store a DataFrame or LazyFrame.
It is just a symbolic reference used to build expressions and frame-dependent recipes.

The empty declaration:
```
__slots__ = ()
```
is appropriate because ``_FrameReference`` does not store any instance attributes.
The singleton ``f`` is stateless: the current frame is passed in later
				  instead of being stored globally inside ``f``.

This matters for nested operations, multiple frames, and concurrent code.
There is no global "current frame" that could accidentally be overwritten.

##----------------------------------------------------------##
## 4. Immediate expressions versus deferred frame selection ##
##----------------------------------------------------------##

``_FrameReference`` provides two different kinds of behavior.

The first kind does not require a concrete frame:
```
def __getitem__(self, name: str) -> pl.Expr:
    return pl.col(name)
```

Therefore:
```
f["x"]
```
immediately becomes:
```
pl.col("x")
```

The same is true for attribute syntax:
```
def __getattr__(self, name: str) -> pl.Expr:
    if name.startswith("_"):
        raise AttributeError(name)

    return pl.col(name)
```

Therefore:
```
f.x
```
also becomes:
```
pl.col("x")
```

A Polars expression can be constructed without knowing its future frame,
so these operations do not need ``_Deferred``.

The situation is different for:
```
f.select("y")
```

A normal ``select()`` belongs to a particular DataFrame or LazyFrame and
returns another frame. Because the current frame is unavailable, ``f.select()``
cannot execute the selection immediately.

Instead, it returns a description of what should happen later:
```
def select(self, *exprs, **named_exprs) -> _Deferred:
    return _Deferred(
        lambda frame: frame.select(*exprs, **named_exprs)
    )
```

This is where ``_FrameReference`` connects to ``_Deferred``.

##------------------------------##
## 5. The idea of deferred work ##
##------------------------------##

When we call:
```
f.select("y")
```

we cannot yet perform:
```
current_frame.select("y")
```

Instead, we store this function:
```
lambda frame: frame.select("y")
```

This function is a recipe with one missing ingredient: ``frame``.

It remembers the requested selection but does nothing
until a DataFrame or LazyFrame is supplied:
```
f.select("y")
    -> store: lambda frame: frame.select("y")
    -> execute later when mutate() supplies frame
```

This delayed execution is called deferral.
The object that stores the recipe is an instance of ``_Deferred``:
```
_Deferred(
    lambda frame: frame.select("y")
)
```

At this stage:

- no column has been selected;
- no DataFrame or LazyFrame has been created;
- no lazy query has been collected;
- only the future operation has been recorded.

##----------------------------------------------------##
## 6. ``_Deferred``: storing and resolving the recipe ##
##----------------------------------------------------##

The constructor receives a function and stores it in ``_resolver``:
```
class _Deferred:
    __slots__ = ("_resolver",)

    def __init__(self, resolver: Callable[[Any], Any]):
        self._resolver = resolver
```

For example:
```
deferred = _Deferred(
    lambda frame: frame.select("y")
)
```

is conceptually equivalent to storing:
```
deferred._resolver = lambda frame: frame.select("y")
```

The annotation:
```
Callable[[Any], Any]
```
means that ``resolver`` is expected to be callable, accept one argument,
and return one value. It documents the interface but does not enforce types at runtime.

The declaration:
```
__slots__ = ("_resolver",)
```
states that a ``_Deferred`` instance only stores ``_resolver``. This can reduce
memory use and prevent accidental additional attributes. It is useful, but it
is not essential to deferred execution itself.

The stored recipe is executed by ``resolve()``:
```
def resolve(self, frame):
    return self._resolver(frame)
```

Therefore:
```
deferred.resolve(current_frame)
```

expands to:
```
current_frame.select("y")
```

Before ``resolve()``, the object contains only a recipe. During ``resolve()``,
the true current frame is supplied and the selection finally runs.

##--------------------------------------------------##
## 7. ``map()`` composes another deferred operation ##
##--------------------------------------------------##

Selecting a column is only the first step.
In our example, the selected frame must then be converted into an Enum expression.

``_Deferred.map()`` adds another operation without executing the existing one:
```
def map(self, function):
    return _Deferred(
        lambda frame: function(self.resolve(frame))
    )
```

Suppose the first deferred object is:
```
d0 = _Deferred(
    lambda frame: frame.select("y")
)
```

We compose the conversion like this:
```
d1 = d0.map(
    lambda selected: as_enum(selected)
)
```

Conceptually, ``d1`` stores:
```
d1 = _Deferred(
    lambda frame: as_enum(
        d0.resolve(frame)
    )
)
```

Its expanded meaning is:
```
lambda frame: as_enum(
    frame.select("y")
)
```

Nothing has executed yet. Execution begins only when we call:
```
d1.resolve(current_frame)
```

Then the steps are:
```
current_frame.select("y")
    -> as_enum(selected_frame)
```

``d0`` and ``d1`` are distinct ``_Deferred`` objects, but they are not
completely independent. The resolver stored by ``d1`` deliberately closes over
``d0`` and calls ``d0.resolve(frame)``.

Their responsibilities are:
```
d0: select the column from the future frame
d1: select the column, then convert it into an Enum expression
```

##--------------------------------------------------##
## 8. How ``as_enum()`` extends the deferred recipe ##
##--------------------------------------------------##

Return to:
```
tf.mutate(
    y=tp.as_enum(f.select("y"))
)
```

Python first evaluates ``f.select("y")``, which returns:
```
d0 = _Deferred(
    lambda frame: frame.select("y")
)
```

Python then calls:
```
tp.as_enum(d0)
```

``as_enum()`` recognizes the deferred input:
```
if isinstance(x, _Deferred):
    return x.map(
        lambda selected: as_enum(
            selected,
            categories=categories,
            reverse=reverse,
        )
    )
```

Because ``x`` is ``d0``, ``x.map(...)`` creates:
```
d1 = _Deferred(
    lambda frame: as_enum(
        d0.resolve(frame),
        categories=categories,
        reverse=reverse,
    )
)
```

Its fully expanded meaning is:
```
d1 = _Deferred(
    lambda frame: as_enum(
        frame.select("y"),
        categories=categories,
        reverse=reverse,
    )
)
```

There is no immediate or infinite recursion. The lambda is stored rather than
executed. Later, the resolved input is a real one-column DataFrame or LazyFrame,
not another ``_Deferred``, so the normal Enum-conversion path runs.

At this point, ``mutate()`` is effectively receiving:
```
tf.mutate(y=d1)
```

But ``d1`` is still a recipe rather than a Polars expression. The mutation
normalization and execution layers must preserve and eventually resolve it.

##---------------------------------------------------------##
## 9. Preserving ``_Deferred`` during mutate normalization ##
##---------------------------------------------------------##

Both ``TibbleFrame.mutate()`` and ``TibbleLazy.mutate()`` normalize their
arguments before calling ``with_columns()``:
```
def mutate(self, *args, over=None, **kwargs):
    exprs = _as_list(args) + _kwargs_as_exprs(kwargs)
    exprs = _over_exprs(exprs, over)

    out = _mutate_cols(self.as_polars(), exprs)
    return out.pipe(_from_polars_frame)
```

For:
```
tf.mutate(y=d1)
```

the keyword arguments initially look like:
```
kwargs = {"y": d1}
```

Normal constant values are converted with ``pl.lit()``. A ``_Deferred`` object
must not be converted into a Polars literal because it represents pending work,
not a data value.

Therefore, ``_lit_expr()`` preserves it:
```
def _lit_expr(x):
    if isinstance(x, _Deferred):
        return x

    if not _is_expr(x):
        x = pl.lit(x)

    return x
```

Without this special case, Polars would attempt:
```
pl.lit(d1)
```

and raise an error because a ``_Deferred`` recipe is not a valid expression literal.

Named mutation arguments also need their output names attached. Therefore,
``_kwargs_as_exprs()`` effectively asks for:
```
d1.alias("y")
```

Because ``d1`` has not produced a Polars expression yet,
the alias operation must also be deferred.

##-------------------------------------------##
## 10. ``_Deferred.alias()`` defers the name ##
##-------------------------------------------##

The ``alias()`` method is:
```
def alias(self, name):
    return self.map(
        lambda expression: expression.alias(name)
    )
```

It uses ``map()`` to add one more layer to the recipe.

Before aliasing:
```
d1(frame)
    = as_enum(
        frame.select("y")
      )
```

Calling:
```
d2 = d1.alias("y")
```

creates:
```
d2(frame)
    = d1.resolve(frame).alias("y")
```

Expanding ``d1`` gives:
```
d2(frame)
    = as_enum(
        frame.select("y")
      ).alias("y")
```

No alias has actually been applied because no Polars expression exists yet.
The recipe only records that the final expression should be named ``"y"``.

After normalization:
```
exprs = [d2]
```

The layers represent:
```
d0: select "y" from the future frame
d1: convert the selected frame into an Enum expression
d2: alias the resulting expression as "y"
```

##------------------------------------------------##
## 11. Final resolution inside ``_mutate_cols()`` ##
##------------------------------------------------##

The deferred recipe finally receives the real working frame inside
``_mutate_cols()``:
```
def _mutate_cols(frame, exprs):
    for expr in exprs:
        if isinstance(expr, _Deferred):
            expr = expr.resolve(frame)

        frame = frame.with_columns(expr)

    return frame
```

Here, ``frame`` is the native DataFrame or LazyFrame produced by:
```
self.as_polars()
```

When the loop reaches ``d2``, this runs:
```
expr = d2.resolve(frame)
```

The recipes unfold from the inside outward:
```
frame.select("y")
    -> as_enum(frame.select("y"))
        -> as_enum(frame.select("y")).alias("y")
```

``as_enum()`` inspects the one-column frame,
determines its column name and categories, and returns approximately:
```
pl.col("y")
  .cast(pl.String)
  .cast(pl.Enum(categories))
```

The outer layer adds:
```
.alias("y")
```

After resolution, ``expr`` is an ordinary Polars expression:
```
pl.col("y")
  .cast(pl.String)
  .cast(pl.Enum(categories))
  .alias("y")
```

It can now be passed safely into:
```
frame.with_columns(expr)
```

Resolving inside the loop also preserves sequential mutation:
```
tf.mutate(
    copied=tp.col("y"),
    ordered=tp.as_enum(f.select("copied")),
)
```

The first expression adds ``copied`` to ``frame``. The next deferred expression
is resolved against that updated frame, so ``f.select("copied")`` can see the
newly created column.

The complete flow is:
```
f.select("y")
    -> deferred selection
    -> deferred Enum conversion
    -> deferred alias
    -> mutate supplies its current frame
    -> resolve all layers
    -> ordinary Polars expression
    -> with_columns()
```

##---------------------------------------##
## 12. Other features of ``f`` namespace ##
##---------------------------------------##

Besides deferred frame selection, ``f`` provides concise column-expression
syntax:
```
f["x"]        # equivalent to pl.col("x")
f.x           # equivalent to pl.col("x")
```

Because these return normal ``pl.Expr`` objects immediately, they work with
Polars operators and supported NumPy universal functions:
```
f["x"] * 2
f.x + f.y
np.sqrt(f["x"])
```

Bracket syntax is safest for unusual names and names that collide with real
``f`` methods:
```
f["column with spaces"]
f["select"]
```

``f.select()`` forwards positional and named selections to the future frame:
```
f.select("x")
f.select("x", doubled=pl.col("y") * 2)
```

The central distinction is:
```
f["x"] or f.x
    -> immediate Polars column expression

f.select("x")
    -> deferred operation requiring the current frame
```

Finally, ``f`` never contains the actual current DataFrame or LazyFrame. It is a
symbolic namespace. Frame-dependent operations receive the current frame only
when a compatible verb, currently ``mutate()``, resolves their ``_Deferred``
recipe.

For a LazyFrame, inferring Enum categories from ``f.select("x")`` requires an
internal collection because the category values must be known. To preserve full
laziness, users should provide categories explicitly:
```
tf.mutate(
    y=tp.as_enum(
        "y",
        categories=["a", "b", "c"],
    )
)
```
"""
