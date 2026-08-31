from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tidypyrs")
except PackageNotFoundError:
    __version__ = "0+unknown"


from . import funs as _funs
from . import lubridate as _lubridate
from . import reexports as _reexports
from . import stringr as _stringr
from . import tibble_frame as _tibble_frame
from . import tibble_lazy as _tibble_lazy
from . import tidyselect as _tidyselect

from .funs import *
from .lubridate import *
from .reexports import *
from .stringr import *
from .tibble_frame import *
from .tibble_lazy import *
from .tidyselect import *


__all__ = [
    "__version__",
    *_funs.__all__,
    *_lubridate.__all__,
    *_reexports.__all__,
    *_stringr.__all__,
    *_tibble_frame.__all__,
    *_tibble_lazy.__all__,
    *_tidyselect.__all__,
]
