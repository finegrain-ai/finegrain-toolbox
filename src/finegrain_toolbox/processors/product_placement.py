import dataclasses as dc

import torch
from PIL import Image

from ..flux import LatentReshaper, Model, Prompt, get_kontext_image_ids
from ..image import draw_bbox, get_actual_bbox, image_resampling, project_bbox, rescale_mod
from ..torch import image_to_tensor, no_grad, tensor_to_image
from ..types import BoundingBox

MAX_SHORT_SIZE = 1024
MAX_LONG_SIZE = 1536


def _padded_size(size: tuple[int, int], padding: float) -> tuple[int, int]:
    return (int(size[0] * (1 + padding)), int(size[1] * (1 + padding)))


def cutout_to_reference_image(
    cutout: Image.Image,
    max_short_size: int,
    max_long_size: int,
) -> Image.Image:
    """
    Turn a cutout into a reference image so that:
    - The shortest side is at most `max_short_size`.
    - The longest side is at most `max_long_size`.
    - The resulting image dimensions are multiples of 64.
    - 2% padding is added on each side.
    - The reference image is not distorted.
    - It is on white background.
    """
    assert max_short_size <= max_long_size, "max_short_size must be less than or equal to max_long_size"
    padding = 0.04

    ref_size = rescale_mod(
        size=_padded_size(cutout.size, padding),
        shortest_side=max_short_size,
        longest_side=max_long_size,
        mod=64,
        only_down=True,
        direction="up",
    )

    resized = cutout.copy()
    resized.thumbnail(_padded_size(ref_size, -padding), image_resampling)

    ref = Image.new(mode="RGB", size=ref_size, color=(255, 255, 255))
    ref.paste(
        resized,
        box=(
            (ref_size[0] - resized.width) // 2,
            (ref_size[1] - resized.height) // 2,
        ),
        mask=resized if resized.mode == "RGBA" else None,
    )

    return ref


@dc.dataclass(kw_only=True)
class Result:
    reference: Image.Image
    scene: Image.Image
    output: Image.Image


@no_grad()
def process(
    model: Model,
    scene: Image.Image,
    reference: Image.Image,
    bbox: BoundingBox,
    prompt: Prompt,
    seed: int = 1234,
    num_steps: int = 28,
    max_short_size: int = MAX_SHORT_SIZE,
    max_long_size: int = MAX_LONG_SIZE,
) -> Result:
    """Insert a product into a scene in a given bounding box.

    Args:
        model: The model to use for blending.
        scene: The scene in which to insert the product.
        reference: The image of the product to insert (RGBA cutout).
        bbox: The bounding box (xmin, ymin, xmax, ymax).
        prompt: The prompt to use ("Add this in the box").
        seed: The random seed.
        num_steps: The number of diffusion steps.
    """
    device, dtype, latent_channels = model.device, model.dtype, model.latent_channels

    assert reference.mode == "RGBA", "reference image must be a RGBA cutout"
    reference = reference.crop(reference.getbbox())  # remove padding

    hr_size = scene.size
    ref_hr_size = reference.size

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    lr_size = rescale_mod(hr_size, max_short_size, max_long_size, 64, only_down=True)

    actual_bbox_hr = get_actual_bbox(bbox, ref_hr_size)
    actual_bbox_lr = project_bbox(actual_bbox_hr, size_in=hr_size, size_out=lr_size)

    scene_hr = scene.convert("RGB")
    scene_lr = scene_hr.resize(lr_size, resample=image_resampling)
    scene_marked = draw_bbox(image=scene_lr, bbox=actual_bbox_lr)

    ref_lr = cutout_to_reference_image(
        cutout=reference,
        max_short_size=max_short_size,
        max_long_size=max_long_size,
    )

    scene_reshaper = LatentReshaper(size=lr_size, latent_channels=latent_channels)
    ref_reshaper = LatentReshaper(size=ref_lr.size, latent_channels=latent_channels)

    x_seq = torch.randn(scene_reshaper.image_sequence_shape, dtype=dtype, generator=generator).to(device)

    image_ids = get_kontext_image_ids(scene_reshaper, [scene_reshaper, ref_reshaper]).to(device, dtype)

    scene_tensor = image_to_tensor(scene_marked, device=device, dtype=dtype)
    scene_latents = model.ae_encode(scene_tensor, generator)
    image_seq = scene_reshaper.pack_image_latents(scene_latents)
    assert image_seq.shape == x_seq.shape

    ref_tensor = image_to_tensor(ref_lr, device=device, dtype=dtype)
    ref_latents = model.ae_encode(ref_tensor, generator)
    ref_seq = ref_reshaper.pack_image_latents(ref_latents)

    timesteps = model.scheduler_set_timesteps(num_steps, scene_reshaper.token_sequence_length)
    guidance = torch.tensor([30], device=device, dtype=torch.float32)

    with torch.inference_mode():
        for t in timesteps:
            assert isinstance(t, torch.Tensor)
            timestep = t.unsqueeze(0).to(dtype) / 1000

            noise_pred = model.transformer(
                hidden_states=torch.cat((x_seq, image_seq, ref_seq), dim=1),
                timestep=timestep,
                guidance=guidance,
                pooled_projections=prompt.clip_embeds,
                encoder_hidden_states=prompt.t5_embeds,
                txt_ids=prompt.text_ids,
                img_ids=image_ids,
                return_dict=False,
            )[0][:, : scene_reshaper.token_sequence_length, :]

            x_seq = model.scheduler.step(noise_pred, t, x_seq, return_dict=False)[0]  # type: ignore

    latents = scene_reshaper.unpack_image_latents(x_seq, prompt.batch_size)

    assert latents.size(0) == 1
    decoded_tensor = model.ae_decode(latents)

    output = tensor_to_image(decoded_tensor).resize(hr_size, resample=image_resampling)
    return Result(reference=ref_lr, scene=scene_marked, output=output)
