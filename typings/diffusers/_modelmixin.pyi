import os

import torch

from finegrain_toolbox.types import Self

class ModelMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        torch_dtype: torch.dtype | None = None,
        subfolder: str | None = None,
        device_map: dict[str, torch.device] | None = None,
        use_safetensors: bool | None = None,
    ) -> Self: ...
    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> None: ...
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True) -> None: ...
    def load_lora_adapter(self, state_dict: dict[str, torch.Tensor], adapter_name: str) -> None: ...
    def set_adapters(self, adapter_names: list[str], weights: list[float]) -> None: ...
    def fuse_lora(self) -> None: ...
    def unload_lora(self) -> None: ...
