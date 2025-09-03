import dataclasses as dc

import torch

# In Flux, pixels are projected into a latent space, then the latents are
# packed into a token sequence.
#
# The latent space has a scaling factor of 8 and 16 channels.
# Image tokens have an embedding length of 64.
#
# Going from the latent space to tokens is a simple reshaping ("packing").
# We go from (B=1, C=16, LH, LW) to (B=1, SL, EL=64) where SL = LH * LW / 4.
# This packing implies pixel-space image dimensions must be multiples of 16.
# (Acronyms: Batch, Channels, Latent Height / Width, Sequence Length, Embedding Length).


@dc.dataclass(kw_only=True)
class LatentReshaper:
    size: tuple[int, int]
    latent_channels: int = 16

    @property
    def pixel_width(self) -> int:
        return self.size[0]

    @property
    def pixel_height(self) -> int:
        return self.size[1]

    @property
    def latent_width(self) -> int:
        return self.pixel_width // 8

    @property
    def latent_height(self) -> int:
        return self.pixel_height // 8

    @property
    def packed_latent_width(self) -> int:
        return self.latent_width // 2

    @property
    def packed_latent_height(self) -> int:
        return self.latent_height // 2

    @property
    def token_sequence_length(self) -> int:
        return self.packed_latent_height * self.packed_latent_width

    @property
    def image_embedding_length(self) -> int:
        return self.latent_channels * 4

    @property
    def mask_embedding_length(self) -> int:
        return self.latent_channels * 16

    @property
    def image_sequence_shape(self) -> tuple[int, int, int]:
        return (1, self.token_sequence_length, self.image_embedding_length)

    @property
    def mask_sequence_shape(self) -> tuple[int, int, int]:
        return (1, self.token_sequence_length, self.mask_embedding_length)

    def pack_image_latents(self, t: torch.Tensor) -> torch.Tensor:
        assert t.shape == (1, self.latent_channels, self.latent_height, self.latent_width)
        t = t.reshape(1, self.latent_channels, self.packed_latent_height, 2, self.packed_latent_width, 2)
        t = t.permute(0, 2, 4, 1, 3, 5)
        t = t.reshape(*self.image_sequence_shape)
        return t

    def pack_mask(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(1, self.packed_latent_height, 2, 8, self.packed_latent_width, 2, 8)
        t = t.permute(0, 1, 4, 3, 6, 2, 5)
        t = t.reshape(*self.mask_sequence_shape)
        return t

    def unpack_image_latents(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        t = t.reshape(batch_size, self.packed_latent_height, self.packed_latent_width, self.latent_channels, 2, 2)
        t = t.permute(0, 3, 1, 4, 2, 5)
        t = t.reshape((batch_size, self.latent_channels, self.latent_height, self.latent_width))
        return t
