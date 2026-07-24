import hashlib
from pathlib import Path


def hash_file(path: str | Path, *, digest_size: int) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
