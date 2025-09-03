import torch

from .latents import LatentReshaper


def get_kontext_image_ids(x_reshaper: LatentReshaper, ref_reshapers: list[LatentReshaper]) -> torch.Tensor:
    ids_list: list[torch.Tensor] = []

    # x: at origin, virtual timestep 0
    x_image_ids = torch.stack(
        [
            torch.zeros(x_reshaper.token_sequence_length),
            torch.arange(x_reshaper.packed_latent_height).repeat_interleave(x_reshaper.packed_latent_width),
            torch.arange(x_reshaper.packed_latent_width).repeat(x_reshaper.packed_latent_height),
        ]
    )
    ids_list.append(torch.transpose(x_image_ids, 0, 1))

    # add scene + all references stacked horizontally
    # scene is at origin, 1st reference is offset by scene width

    offset = 0
    for i, reshaper in enumerate(ref_reshapers):
        virtual_timestep = i + 1
        image_ids = torch.stack(
            [
                torch.ones(reshaper.token_sequence_length) * virtual_timestep,
                torch.arange(reshaper.packed_latent_height).repeat_interleave(reshaper.packed_latent_width),
                torch.arange(
                    offset,
                    offset + reshaper.packed_latent_width,
                ).repeat(reshaper.packed_latent_height),
            ]
        )
        ids_list.append(torch.transpose(image_ids, 0, 1))
        offset += reshaper.packed_latent_width

    return torch.cat(ids_list, dim=0)
