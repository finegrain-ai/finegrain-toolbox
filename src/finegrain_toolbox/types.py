import sys
from typing import Any, cast

Size2D = tuple[int, int]
BoundingBox = tuple[int, int, int, int]


def as_size2d(v: Any) -> Size2D:
    if isinstance(v, str):
        r = tuple(int(x) for x in v.split("x"))
    elif isinstance(v, list):
        v = cast(list[Any], v)
        assert all(isinstance(x, int) for x in v)
        r = cast(tuple[int, ...], tuple(v))
    else:
        assert isinstance(v, tuple)
        v = cast(tuple[Any, ...], v)
        assert all(isinstance(x, int) for x in v)
        r = cast(tuple[int, ...], v)
    assert len(r) == 2
    return r


def as_bbox(v: Any) -> BoundingBox:
    if isinstance(v, str):
        r = tuple(int(x) for x in v.split(","))
    elif isinstance(v, list):
        v = cast(list[Any], v)
        assert all(isinstance(x, int) for x in v)
        r = cast(tuple[int, ...], tuple(v))
    else:
        assert isinstance(v, tuple)
        v = cast(tuple[Any, ...], v)
        assert all(isinstance(x, int) for x in v)
        r = cast(tuple[int, ...], v)
    assert len(r) == 4
    return r


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


if sys.version_info < (3, 12):
    from typing_extensions import Buffer
else:
    from collections.abc import Buffer

__all__ = ["Self", "Buffer"]
