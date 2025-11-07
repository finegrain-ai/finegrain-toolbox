import dataclasses as dc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def shallow_asdict(m: "DataclassInstance") -> dict[str, Any]:
    return {field.name: getattr(m, field.name) for field in dc.fields(m)}


class DcMixin:
    def shallow_asdict(self) -> dict[str, Any]:
        assert dc.is_dataclass(self)
        return shallow_asdict(self)
