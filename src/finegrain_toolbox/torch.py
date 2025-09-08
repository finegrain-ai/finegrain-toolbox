# This file contains well-typed version of torch utils and a few helpers.

from typing import Any

import torch
from numpy import array, float32
from PIL import Image
from torch import Tensor

if torch.cuda.is_available():
    default_device = torch.device("cuda:0")
    default_dtype = torch.bfloat16
else:
    default_device = torch.device("cpu")
    default_dtype = torch.float32


def manual_seed(seed: int) -> None:
    torch.manual_seed(seed)  # type: ignore


class no_grad(torch.no_grad):
    def __new__(cls, orig_func: Any | None = None) -> "no_grad":  # type: ignore
        return object.__new__(cls)


def image_to_tensor(
    image: Image.Image,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Convert a PIL Image to a Tensor.

    Args:
        image: The image to convert.
        device: The device to use for the tensor.
        dtype: The dtype to use for the tensor.

    Returns:
        The converted tensor.

    Note:
        If the image is in mode `RGB` the tensor will have shape `[3, H, W]`,
        otherwise `[1, H, W]` for mode `L` (grayscale) or `[4, H, W]` for mode `RGBA`.

        Values are normalized to the range `[0, 1]`.
    """
    image_tensor = torch.tensor(array(image).astype(float32) / 255.0, device=device, dtype=dtype)

    assert isinstance(image.mode, str)  # type: ignore
    match image.mode:
        case "L":
            image_tensor = image_tensor.unsqueeze(0)
        case "RGBA" | "RGB":
            image_tensor = image_tensor.permute(2, 0, 1)
        case _:
            raise ValueError(f"Unsupported image mode: {image.mode}")

    return image_tensor.unsqueeze(0)


def tensor_to_image(tensor: Tensor) -> Image.Image:
    """Convert a Tensor to a PIL Image.

    Args:
        tensor: The tensor to convert.

    Returns:
        The converted image.

    Note:
        The tensor must have shape `[1, channels, height, width]` where the number of
        channels is either 1 (grayscale) or 3 (RGB) or 4 (RGBA).

        Expected values are in the range `[0, 1]` and are clamped to this range.
    """
    assert tensor.ndim == 4 and tensor.shape[0] == 1, f"Unsupported tensor shape: {tensor.shape}"
    num_channels = tensor.shape[1]
    tensor = tensor.clamp(0, 1).squeeze(0)
    tensor = tensor.to(torch.float32)  # to avoid numpy error with bfloat16

    match num_channels:
        case 1:
            tensor = tensor.squeeze(0)
        case 3 | 4:
            tensor = tensor.permute(1, 2, 0)
        case _:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

    return Image.fromarray((tensor.cpu().numpy() * 255).astype("uint8"))  # type: ignore[reportUnknownType]
