import torch

from ._modelmixin import ModelMixin

class FluxLoraLoaderMixin:
    @classmethod
    def _maybe_expand_lora_state_dict(
        cls,
        transformer: ModelMixin,
        lora_state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]: ...
