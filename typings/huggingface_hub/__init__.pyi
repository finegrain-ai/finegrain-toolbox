from pathlib import Path
from typing import Any

def hf_hub_download(repo_id: str, filename: str) -> str: ...

# actual return type is CommitInfo
def upload_folder(
    repo_id: str,
    folder_path: str | Path,
    commit_message: str | None = None,
    token: str | None = None,
    create_pr: bool | None = None,
) -> Any: ...
