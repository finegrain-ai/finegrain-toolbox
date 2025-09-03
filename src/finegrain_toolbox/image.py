import math
from typing import Literal

from PIL import Image, ImageDraw

from .types import BoundingBox, Size2D

RoundDirection = Literal["up", "down", "round"]

image_resampling: Image.Resampling = Image.Resampling.BICUBIC
mask_resampling: Image.Resampling = Image.Resampling.BICUBIC


def _fit_mod(n: int, bound: int, mod: int, direction: RoundDirection = "round") -> int:
    match direction:
        case "down":
            pad = 0
        case "up":
            pad = mod - 1
        case "round":
            pad = mod // 2
    return max(min(mod * ((n + pad) // mod), bound), mod)


def rescale_mod(
    size: Size2D,
    shortest_side: int,
    longest_side: int,
    mod: int,
    *,
    only_down: bool,
    direction: RoundDirection = "round",
) -> tuple[int, int]:
    # resize to fit a shortest and longest side and so both dimensions are a multiple of `mod`
    assert shortest_side % mod == 0, f"shortest_side ({shortest_side}) must be a multiple of mod ({mod})"
    assert longest_side % mod == 0, f"longest_side ({longest_side}) must be a multiple of mod ({mod})"
    short_in, long_in = min(size), max(size)
    if (long_in / short_in) <= (longest_side / shortest_side):  # resize based on shortest side
        if only_down and short_in < shortest_side:
            shortest_side = _fit_mod(short_in, shortest_side, mod, direction)
        longest_side = _fit_mod(shortest_side * long_in // short_in, longest_side, mod, direction)
    else:
        if only_down and long_in < longest_side:
            longest_side = _fit_mod(long_in, longest_side, mod, direction)
        shortest_side = _fit_mod(longest_side * short_in // long_in, shortest_side, mod, direction)
    return (shortest_side, longest_side) if size[0] < size[1] else (longest_side, shortest_side)


def project_bbox(bbox_in: BoundingBox, size_in: Size2D, size_out: Size2D) -> BoundingBox:
    return (
        math.floor(bbox_in[0] * size_out[0] / size_in[0]),
        math.floor(bbox_in[1] * size_out[1] / size_in[1]),
        math.ceil(bbox_in[2] * size_out[0] / size_in[0]),
        math.ceil(bbox_in[3] * size_out[1] / size_in[1]),
    )


def get_actual_bbox(target_bbox: BoundingBox, subject_size: Size2D) -> BoundingBox:
    target_bbox_w = target_bbox[2] - target_bbox[0]
    target_bbox_h = target_bbox[3] - target_bbox[1]
    subject_w, subject_h = subject_size

    # calculate scaling ratios for both width and height
    width_ratio = target_bbox_w / subject_w
    height_ratio = target_bbox_h / subject_h

    # use the smaller ratio to ensure the subject fits within target bounds
    ratio = min(width_ratio, height_ratio)
    bbox_w, bbox_h = ratio * subject_w, ratio * subject_h

    # compute the bbox coordinates from its width and height
    # center horizontally within target bbox
    x1 = target_bbox[0] + (target_bbox_w - bbox_w) // 2
    # position at bottom of target bbox
    y1 = target_bbox[3] - bbox_h
    x2 = x1 + bbox_w
    y2 = target_bbox[3]

    return (int(x1), int(y1), int(x2), int(y2))


def draw_bbox(
    image: Image.Image,
    bbox: BoundingBox,
    color: str = "red",
    thickness: int = 6,
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=color, width=thickness)
    return image
