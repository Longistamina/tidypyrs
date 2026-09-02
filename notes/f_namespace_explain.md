\# =========================================================================   
\# EXPLANATION OF F_NAMESPACE   
\# =========================================================================   

\##-------------------------------------------   
\## 1. The goal of this file   
\##-------------------------------------------   

Most Tidypyrs functions work directly with Polars expressions:
```
tf.mutate(doubled=tp.col("x") * 2)
```

However, functions such as ``tp.as_enum()`` and ``tp.as_ordered()`` can ONLY infer
their categories from a single-column DataFrame or LazyFrame, not merely from a
column expression. For example, it only works in similar codes like this one:
```python
tp.as_enum(tf.select("y"))
```

This is inconvenient inside a method chain because the current frame must be
named before we can select from it. Therefore, during a method chaining workflow,
we have to call `.pipe()` method:
```python
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
```python
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

\##---------------------------------------------------------------------------   
\## 2. Why the current frame is unavailable   
\##---------------------------------------------------------------------------   

Consider:
```python
tl.mutate(
    y=tp.as_enum(f.select("y"))
)
```

Before Python can execute the body of ``tl.mutate()``, it must evaluate the
arguments passed to the method. The relevant flow is:
```python
f.select("y")
    -> tp.as_enum(...)
        -> tl.mutate(y=...)
```

This is not a general LIFO or stack-memory rule. It follows from expression
dependencies: Python needs the result of ``f.select("y")`` before it can call
``tp.as_enum()``, and it needs the result of ``tp.as_enum()`` before it can call
the body of ``tl.mutate()``.

Therefore, when ``f.select("y")`` runs, ``mutate()`` has not yet supplied its
current DataFrame or LazyFrame.

If ``f`` tried to perform a normal selection immediately, Python would need to
know which concrete frame this means:
```python
f.select("y")
```

But no frame has been passed to ``f``. We first need a real, defined ``f``
object, and then a way for it to describe a selection whose frame will only
become known later.

\##------------------------------------------------------------------------------------------------   
\## 3. ``_FrameReference`` defines the public ``f`` namespace   
\##------------------------------------------------------------------------------------------------   

The first problem is that ``f`` must be an actual Python object with known
behavior. This is solved by ``_FrameReference``:
```python
class _FrameReference:
    __slots__ = ()

    ...

f = _FrameReference()
```

The class defines what operations such as ``f["x"]``, ``f.x``, and
``f.select("x")`` mean. The final line creates one public instance named ``f``.

After this assignment, ``f`` is no longer unknown or undefined. Python knows
that it is an instance of ``_FrameReference`` and can look up the behavior
defined by that class.

However, ``f`` does not store a DataFrame or LazyFrame. It is a symbolic
reference used to build expressions and frame-dependent recipes.

The empty declaration:
```python
__slots__ = ()
```
is appropriate because ``_FrameReference`` does not store any instance
attributes. The singleton ``f`` is stateless: the current frame is passed in
later instead of being stored globally inside ``f``.

This matters for nested operations, multiple frames, and concurrent code.
There is no global "current frame" that could accidentally be overwritten.

\##-----------------------------------------------------------------------------------------------------   
\## 4. Immediate expressions versus deferred frame selection   
\##-----------------------------------------------------------------------------------------------------   

``_FrameReference`` provides two different kinds of behavior.

The first kind does not require a concrete frame:
```python
def __getitem__(self, *names: str) -> pl.Expr:
    return pl.col(*names)
```

Therefore:
```python
f["x"]
```
immediately becomes:
```python
pl.col("x")
```

The same is true for attribute syntax:
```python
def __getattr__(self, name: str) -> pl.Expr:
    if name.startswith("_"):
        raise AttributeError(name)

    return pl.col(name)
```

Therefore:
```python
f.x
```
also becomes:
```python
pl.col("x")
```

A Polars expression can be constructed without knowing its future frame, so
these operations do not need ``_Deferred``.

The situation is different for:
```python
f.select("y")
```

A normal ``select()`` belongs to a particular DataFrame or LazyFrame and
returns another frame. Because the current frame is unavailable, ``f.select()``
cannot execute the selection immediately.

Instead, it returns a description of what should happen later:
```python
def select(self, *exprs, **named_exprs) -> _Deferred:
    return _Deferred(
        lambda frame: frame.select(*exprs, **named_exprs)
    )
```

This is where ``_FrameReference`` connects to ``_Deferred``.

\##----------------------------------------------------------------------------------   
\## 5. The idea of deferred work   
\##-----------------------------------------------------------------------------------   

When we call:
```python
f.select("y")
```

we cannot yet perform:
```python
current_frame.select("y")
```

Instead, we store this function:
```python
lambda frame: frame.select("y")
```

This function is a recipe with one missing ingredient: ``frame``.

It remembers the requested selection but does nothing until a DataFrame or
LazyFrame is supplied:
```python
f.select("y")
    -> store: lambda frame: frame.select("y")
    -> execute later when mutate() supplies frame
```

This delayed execution is called deferral. The object that stores the recipe is
an instance of ``_Deferred``:
```python
_Deferred(
    lambda frame: frame.select("y")
)
```

At this stage:

- no column has been selected;
- no DataFrame or LazyFrame has been created;
- no lazy query has been collected;
- only the future operation has been recorded.

\##-----------------------------------------------------------------------------------------------------   
\## 6. ``_Deferred``: storing and resolving the recipe   
\##-----------------------------------------------------------------------------------------------------   

The constructor receives a function and stores it in ``_resolver``:
```python
class _Deferred:
    __slots__ = ("_resolver",)

    def __init__(self, resolver: Callable[[Any], Any]):
        self._resolver = resolver
```

For example:
```python
deferred = _Deferred(
    lambda frame: frame.select("y")
)
```

is conceptually equivalent to storing:
```python
deferred._resolver = lambda frame: frame.select("y")
```

The annotation:
```python
Callable[[Any], Any]
```
means that ``resolver`` is expected to be callable, accept one argument, and
return one value. It documents the interface but does not enforce types at
runtime.

The declaration:
```python
__slots__ = ("_resolver",)
```
states that a ``_Deferred`` instance only stores ``_resolver``. This can reduce
memory use and prevent accidental additional attributes. It is useful, but it
is not essential to deferred execution itself.

The stored recipe is executed by ``resolve()``:
```python
def resolve(self, frame):
    return self._resolver(frame)
```

Therefore:
```python
deferred.resolve(current_frame)
```

expands to:
```python
current_frame.select("y")
```

Before ``resolve()``, the object contains only a recipe. During ``resolve()``,
the true current frame is supplied and the selection finally runs.

\##-----------------------------------------------------------------------------------------------------   
\## 7. ``map()`` composes another deferred operation   
\##-----------------------------------------------------------------------------------------------------   

Selecting a column is only the first step. In our example, the selected frame
must later be passed into ``as_enum()``.

``_Deferred.map()`` adds another operation without executing the existing one:
```python
def map(self, function):
    return _Deferred(
        lambda frame: function(self.resolve(frame))
    )
```

Suppose the first deferred object is:
```python
d0 = _Deferred(
    lambda frame: frame.select("y")
)
```

In this demonstration, let's use ``as_enum()``:
```python
d1 = d0.map(
    lambda selected: as_enum(selected)
)
```

Conceptually, ``d1`` stores:
```python
d1 = _Deferred(
    lambda frame: as_enum(
        d0.resolve(frame)
    )
)
```

Its expanded meaning is:
```python
lambda frame: as_enum(
    frame.select("y")
)
```

Nothing has executed yet. Execution begins only when we call:
```python
d1.resolve(current_frame)
```

Then the steps are:
```python
current_frame.select("y")
    -> as_enum(current_frame.select("y"))
```

``d0`` and ``d1`` are distinct ``_Deferred`` objects, but they are not
completely independent. The resolver stored by ``d1`` deliberately closes over
``d0`` and calls ``d0.resolve(frame)``.

Their responsibilities are:
```
d0: select the column from the future frame
d1: select the column, then convert it into an Enum expression
```

\##-----------------------------------------------------------------------------------------------------   
\## 8. ``functools.wraps``, ``_defer_aware``, and ``as_enum()``   
\##-----------------------------------------------------------------------------------------------------   

We want ``as_enum()`` to support both of these inputs:
```python
tp.as_enum(tf.select("y"))  # an actual one-column frame
tp.as_enum(f.select("y"))   # a deferred one-column selection
```

Without changing every line inside ``as_enum()``, we can place the deferred
dispatching behavior in a reusable decorator named ``_defer_aware``:
```python
from functools import wraps

def _defer_aware(function):
    @wraps(function)
    def wrapper(x, *args, **kwargs):
        if isinstance(x, _Deferred):
            return x.map(
                lambda resolved: function(
                    resolved,
                    *args,
                    **kwargs,
                )
            )

        return function(x, *args, **kwargs)

    return wrapper
```

### What makes ``_defer_aware`` a decorator?

A Python decorator is a callable that receives a function and returns the
function that should replace it. ``_defer_aware`` has exactly that structure:
```python
def _defer_aware(function):
    ...
    return wrapper
```

Here, ``function`` is the original function being decorated. ``wrapper`` is the
new function that adds deferred-input handling before deciding when to call the
original function.

Applying the decorator with ``@`` syntax:
```python
@_defer_aware
def as_enum(x, categories=None, reverse=False):
    ...
```

is equivalent to:
```python
def as_enum(x, categories=None, reverse=False):
    ...

as_enum = _defer_aware(as_enum)
# |                      |
# v                      v
# wrapper             function = original_as_enum
```

After decoration, the module-level name ``as_enum`` refers to ``wrapper``. The
``function`` variable captured inside ``wrapper`` still refers to the original,
undecorated ``as_enum`` implementation (let's call it ``orginal_as_enum``). This is why the decorator can call:
```python
return function(x, *args, **kwargs)
```

without recursively calling the wrapper again.

### What does ``functools.wraps`` do?

Inside ``_defer_aware``, ``@wraps(function)`` decorates ``wrapper``:
```python
@wraps(function)
def wrapper(x, *args, **kwargs):
    ...
```

This is approximately equivalent to:
```python
def wrapper(x, *args, **kwargs):
    ...

wrapper = wraps(function)(wrapper)
```

``wraps`` copies important metadata from the original function to the wrapper,
including its name, module, annotations, and docstring. It also sets
``wrapper.__wrapped__`` to the original function. Consequently, documentation
tools, ``help()``, debuggers, and introspection tools continue to see the public
function as ``as_enum`` rather than as a generic function named ``wrapper``.

Importantly, ``wraps`` does not make ``_defer_aware`` a decorator.
``_defer_aware`` is already a decorator because it accepts ``function`` and
returns ``wrapper``. ``wraps`` only preserves the decorated function's identity
and metadata.

### Applying ``_defer_aware`` to ``as_enum()``

The function is now written normally, with no deferred branch inside its body:
```python
@_defer_aware
def as_enum(x, categories=None, reverse=False):
    if categories is None:
        if isinstance(x, pl.DataFrame):
            ...
        elif isinstance(x, pl.LazyFrame):
            ...
        else:
            raise ValueError(...)

    ...

    return (
        pl.col(x)
        .cast(pl.String)
        .cast(pl.Enum(categories))
    )
```

For a normal input:
```python
tp.as_enum(tf.select("y"))
```

the public name ``as_enum`` calls ``wrapper``. Since ``x`` is not deferred, the
wrapper immediately calls the original function:
```python
return function(x, *args, **kwargs)
```

For a deferred input, Python first evaluates:
```python
d0 = f.select("y")
```

and then calls:
```python
tp.as_enum(d0)
```

The wrapper detects ``d0`` and returns:
```python
d1 = d0.map(
    lambda resolved: function(
        resolved,
        categories=categories,
        reverse=reverse,
    )
)
```

Conceptually, ``d1`` stores:
```python
d1 = _Deferred(
    lambda frame: original_as_enum(
        frame.select("y"),
        categories=categories,
        reverse=reverse,
    )
)
```

There is no immediate execution and no infinite recursion. Later,
``d1.resolve(frame)`` first produces the real one-column frame and then passes
it directly into the original ``as_enum()`` implementation.

At this point, ``mutate()`` is effectively receiving:
```python
tl.mutate(y=d1)
```

But ``d1`` is still a recipe rather than a Polars expression. The mutation
normalization and execution layers must preserve and eventually resolve it.

\##-----------------------------------------------------------------------------------------------------   
\## 9. Preserving ``_Deferred`` during mutate normalization   
\##-----------------------------------------------------------------------------------------------------   

Both ``TibbleFrame.mutate()`` and ``TibbleLazy.mutate()`` normalize their
arguments before calling ``with_columns()``:
```python
def mutate(self, *args, over=None, **kwargs):
    exprs = _as_list(args) + _kwargs_as_exprs(kwargs)
    exprs = _over_exprs(exprs, over)

    out = _mutate_cols(self.as_polars(), exprs)
    return out.pipe(_from_polars_frame)
```

For:
```python
tl.mutate(y=d1)
```

the keyword arguments initially look like:
```python
kwargs = {"y": d1}
```

Normal constant values are converted with ``pl.lit()``. A ``_Deferred`` object
must not be converted into a Polars literal because it represents pending work,
not a data value.

Therefore, ``_lit_expr()`` preserves it:
```python
def _lit_expr(x):
    if isinstance(x, _Deferred):
        return x

    if not _is_expr(x):
        x = pl.lit(x)

    return x
```

Without this special case, Polars would attempt:
```python
pl.lit(d1)
```

and raise an error because a ``_Deferred`` recipe is not a valid expression
literal.

Named mutation arguments also need their output names attached. Therefore,
``_kwargs_as_exprs()`` effectively asks for:
```python
d1.alias("y")
```

Because ``d1`` has not produced a Polars expression yet, the alias operation
must also be deferred.

\##--------------------------------------------------------------------------------------------   
\## 10. ``_Deferred.alias()`` defers the name   
\##--------------------------------------------------------------------------------------------   

The ``alias()`` method is:
```python
def alias(self, name):
    return self.map(
        lambda expression: expression.alias(name)
    )
```

It uses ``map()`` to add one more layer to the recipe.

Before aliasing:
```python
d1(frame)
    = original_as_enum(
        frame.select("y")
      )
```

Calling:
```python
d2 = d1.alias("y")
```

creates:
```python
d2(frame)
    = d1.resolve(frame).alias("y")
```

Expanding ``d1`` gives:
```python
d2(frame)
    = original_as_enum(
        frame.select("y")
      ).alias("y")
```

No alias has actually been applied because no Polars expression exists yet.
The recipe only records that the final expression should be named ``"y"``.

After normalization:
```python
exprs = [d2]
```

The layers represent:
```
d0: select "y" from the future frame
d1: convert the selected frame into an Enum expression
d2: alias the resulting expression as "y"
```

\##-----------------------------------------------------------------------------------------------   
\## 11. Final resolution inside ``_mutate_cols()``   
\##-----------------------------------------------------------------------------------------------   

The deferred recipe finally receives the real working frame inside
``_mutate_cols()``:
```python
def _mutate_cols(frame, exprs):
    for expr in exprs:
        if isinstance(expr, _Deferred):
            expr = expr.resolve(frame)

        frame = frame.with_columns(expr)

    return frame
```

Here, ``frame`` is the native DataFrame or LazyFrame produced by:
```python
self.as_polars()
```

When the loop reaches ``d2``, this runs:
```python
expr = d2.resolve(frame)
```

The recipes unfold from the inside outward:
```python
frame.select("y")
    -> original_as_enum(frame.select("y"))
        -> as_enum(frame.select("y")).alias("y")
```

The original ``as_enum()`` implementation inspects the one-column frame,
determines its column name and categories, and returns approximately:
```python
pl.col("y")
  .cast(pl.String)
  .cast(pl.Enum(categories))
```

The outer layer adds:
```python
.alias("y")
```

After resolution, ``expr`` is an ordinary Polars expression:
```python
pl.col("y")
  .cast(pl.String)
  .cast(pl.Enum(categories))
  .alias("y")
```

It can now be passed safely into:
```python
frame.with_columns(expr)
```

Resolving inside the loop also preserves sequential mutation:
```python
tf.mutate(
    copied=tp.col("y"),
    ordered=tp.as_enum(f.select("copied")),
)
```

The first expression adds ``copied`` to ``frame``. The next deferred expression
is resolved against that updated frame, so ``f.select("copied")`` can see the
newly created column.

The complete flow is:
```python
f.select("y")
    -> deferred selection
    -> deferred Enum conversion
    -> deferred alias
    -> mutate supplies its current frame
    -> resolve all layers
    -> ordinary Polars expression
    -> with_columns()
```

\##---------------------------------------------------------------------------------------   
\## 12. Other features of ``f`` namespace   
\##---------------------------------------------------------------------------------------   

Besides deferred frame selection, ``f`` provides concise column-expression
syntax:
```python
f["x"]        # equivalent to pl.col("x")
f["x", "y"]   # equivalent to pl.col("x", "y")

f("x")        # equivalent to pl.col("x")
f("x", "y")   # equivalent to pl.col("x", "y")

f.x           # equivalent to pl.col("x")
```

Because these return normal ``pl.Expr`` objects immediately, they work with
Polars operators and supported NumPy universal functions:
```python
f["x"] * 2
f.x + f.y
np.sqrt(f["x"])
```

Bracket syntax is safest for unusual names and names that collide with real
``f`` methods:
```python
f["column with spaces"]
f["select"]
```

``f.select()`` forwards positional and named selections to the future frame:
```python
f.select("x")
f.select("x", doubled=pl.col("y") * 2)
```

The central distinction is:
```python
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
```python
tl.mutate(
    y=tp.as_enum(
        "y",
        categories=["a", "b", "c"],
    )
)
```
