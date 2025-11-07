import os
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from finegrain_toolbox.types import Self

from ._modelmixin import ModelMixin

class ConfigMixin:
    config: dict[str, Any]

class DiagonalGaussianDistribution:
    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor: ...
    def mode(self) -> torch.Tensor: ...

class AutoencoderKL(ModelMixin, ConfigMixin):
    # *always* pass return_dict=False to encode and decode
    def encode(self, x: torch.Tensor, return_dict: bool) -> tuple[DiagonalGaussianDistribution]: ...
    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...

class FluxTransformer2DModel(ModelMixin):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int | None = None,
        guidance_embeds: bool = False,
    ) -> None: ...
    # *always* pass return_dict=False to __call__
    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor,
        return_dict: bool,
    ) -> tuple[torch.Tensor]: ...

class FlowMatchEulerDiscreteScheduler(ModelMixin, ConfigMixin):
    timesteps: torch.Tensor

    def set_timesteps(
        self,
        device: torch.device | None = None,
        sigmas: list[float] | npt.NDArray[np.float32] | None = None,
        mu: float | None = None,
    ) -> None: ...

    # *always* pass return_dict=False to step
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool,
    ) -> tuple[torch.Tensor]: ...

class SchedulerMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        subfolder: str | None = None,
    ) -> Self: ...
    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        push_to_hub: bool = False,
    ) -> None: ...

class DPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    timesteps: torch.Tensor

    def set_timesteps(self, num_inference_steps: int) -> None: ...

    # *always* pass return_dict=False to step
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor]: ...

class UNet2DConditionModel(ModelMixin, ConfigMixin):
    def __init__(self, cross_attention_dim: int | tuple[int, ...] = 1280) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        torch_dtype: torch.dtype | None = None,
        subfolder: str | None = None,
        device_map: dict[str, torch.device] | None = None,
        use_safetensors: bool | None = None,
    ) -> Self: ...
    # *always* pass return_dict=False to __call__
    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor]: ...
