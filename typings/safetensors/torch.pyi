import os

import torch

def save_file(
    tensors: dict[str, torch.Tensor],
    filename: str | os.PathLike[str],
    metadata: dict[str, str] | None = None,
) -> None: ...
def load_file(
    filename: str | os.PathLike[str],
    device: str = "cpu",
) -> dict[str, torch.Tensor]: ...
