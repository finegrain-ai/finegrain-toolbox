from pathlib import Path

from huggingface_hub import hf_hub_download


def resolve_local_path(path: str) -> Path:
    """Resolve a path or `hf://org/repo/path` URI to a local file path (downloading if needed)."""
    if path.startswith("hf://"):
        parts = path.removeprefix("hf://").split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"expected hf://org/repo/path, got {path}")
        repo_id = f"{parts[0]}/{parts[1]}"
        return Path(hf_hub_download(repo_id=repo_id, filename=parts[2]))
    return Path(path)
