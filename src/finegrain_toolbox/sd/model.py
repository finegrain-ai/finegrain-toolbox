import dataclasses as dc
import os
import pathlib
from typing import Any

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel

from ..dc import DcMixin
from ..models import SafePushToHubMixin
from ..torch import default_device, default_dtype
from ..types import Self


@dc.dataclass(kw_only=True)
class WithDPMSolver:
    scheduler: DPMSolverMultistepScheduler

    @property
    def timesteps(self) -> torch.Tensor:
        return self.scheduler.timesteps

    def scheduler_set_timesteps(self, num_steps: int) -> torch.Tensor:
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps
        assert len(timesteps) == num_steps
        return timesteps

    def scheduler_step(
        self,
        model_output: torch.Tensor,
        step: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        return self.scheduler.step(
            model_output=model_output,
            timestep=self.timesteps[step],
            sample=sample,
            return_dict=False,
        )[0]


@dc.dataclass(kw_only=True)
class WithAutoencoderKL:
    autoencoder: AutoencoderKL
    vae_scaling_factor: float = 0.18215
    downscale_factor: int = 8
    latent_channels: int = 4

    def ae_encode(self, image_tensor: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        latents = self.autoencoder.encode(2 * image_tensor - 1, return_dict=False)[0]
        if generator is None:
            latents = latents.mode()  # same as latents.mean
        else:
            latents = latents.sample(generator)
        return latents * self.vae_scaling_factor

    def ae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae_scaling_factor
        image_tensor = self.autoencoder.decode(latents, return_dict=False)[0]
        return ((image_tensor + 1) / 2).clamp(0, 1)


@dc.dataclass(kw_only=True)
class Model(SafePushToHubMixin, WithAutoencoderKL, WithDPMSolver, DcMixin):
    device: torch.device
    dtype: torch.dtype
    unet: UNet2DConditionModel

    def unet_step(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        params = self.shallow_asdict()
        if device is not None:
            params["device"] = device
        if dtype is not None:
            params["dtype"] = dtype
        params["unet"] = self.unet.to(device=device, dtype=dtype)
        params["autoencoder"] = self.autoencoder.to(device=device, dtype=dtype)
        return self.__class__(**params)

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str | pathlib.Path,
        device: torch.device = default_device,
        dtype: torch.dtype = default_dtype,
        use_safetensors: bool | None = None,
    ) -> Self:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            path_or_id,
            subfolder="scheduler",
        )

        autoencoder = AutoencoderKL.from_pretrained(
            path_or_id,
            subfolder="vae",
            use_safetensors=use_safetensors,
            torch_dtype=dtype,
        ).to(device)
        unet = UNet2DConditionModel.from_pretrained(
            path_or_id,
            subfolder="unet",
            torch_dtype=dtype,
            use_safetensors=use_safetensors,
        ).to(device)

        return cls(
            device=device,
            dtype=dtype,
            scheduler=scheduler,
            autoencoder=autoencoder,
            unet=unet,
        )

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        *,
        safe_serialization: bool = False,
        **kwargs: dict[str, Any],
    ):
        self.autoencoder.save_pretrained(
            os.path.join(save_directory, "vae"),
            safe_serialization=safe_serialization,
        )
        self.unet.save_pretrained(
            os.path.join(save_directory, "unet"),
            safe_serialization=safe_serialization,
        )
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))

    def push_to_hub(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
        raise RuntimeError("use safe_push_to_hub instead")
