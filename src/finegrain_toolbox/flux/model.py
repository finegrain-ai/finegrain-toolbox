import dataclasses as dc
import pathlib

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.utils.hub_utils import PushToHubMixin

from ..torch import default_device, default_dtype
from ..types import Self


def get_mu(scheduler: FlowMatchEulerDiscreteScheduler, image_seq_len: int) -> float:
    config = scheduler.config
    max_shift = config["max_shift"]
    base_shift = config["base_shift"]
    max_seq_len = config["max_image_seq_len"]  # 4096
    base_seq_len = config["base_image_seq_len"]  # 256

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@dc.dataclass(kw_only=True)
class Model(PushToHubMixin):
    device: torch.device
    dtype: torch.dtype
    scheduler: FlowMatchEulerDiscreteScheduler
    autoencoder: AutoencoderKL
    transformer: FluxTransformer2DModel
    vae_scaling_factor: float
    vae_shift_factor: float
    downscale_factor: int = 8
    latent_channels: int = 16

    def ae_encode(self, image_tensor: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        latents = self.autoencoder.encode(2 * image_tensor - 1, return_dict=False)[0]
        if generator is None:
            latents = latents.mode()  # same as latents.mean
        else:
            latents = latents.sample(generator)
        return (latents - self.vae_shift_factor) * self.vae_scaling_factor

    def ae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae_scaling_factor + self.vae_shift_factor
        image_tensor = self.autoencoder.decode(latents, return_dict=False)[0]
        return ((image_tensor + 1) / 2).clamp(0, 1)

    def scheduler_set_timesteps(self, num_steps: int, sequence_length: int) -> torch.Tensor:
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps, dtype=np.float32)
        mu = get_mu(self.scheduler, sequence_length)
        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)
        timesteps = self.scheduler.timesteps
        assert len(timesteps) == num_steps
        return timesteps

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        params = dc.asdict(self)
        if device is not None:
            params["device"] = device
        if dtype is not None:
            params["dtype"] = dtype
        params["scheduler"] = self.scheduler.to(device=device, dtype=dtype)
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
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            path_or_id,
            subfolder="scheduler",
        )

        autoencoder = AutoencoderKL.from_pretrained(
            path_or_id,
            subfolder="vae",
            use_safetensors=use_safetensors,
            torch_dtype=dtype,
        ).to(device)
        transformer = FluxTransformer2DModel.from_pretrained(
            path_or_id,
            subfolder="transformer",
            torch_dtype=dtype,
            use_safetensors=use_safetensors,
        ).to(device)

        vae_shift_factor = autoencoder.config["shift_factor"]
        vae_scaling_factor = autoencoder.config["scaling_factor"]
        assert isinstance(vae_shift_factor, float)
        assert isinstance(vae_scaling_factor, float)

        return cls(
            device=device,
            dtype=dtype,
            scheduler=scheduler,
            autoencoder=autoencoder,
            transformer=transformer,
            vae_scaling_factor=vae_scaling_factor,
            vae_shift_factor=vae_shift_factor,
        )
