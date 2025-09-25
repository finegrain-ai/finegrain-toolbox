import abc
import tempfile
from typing import Any

from huggingface_hub import upload_folder


class SafePushToHubMixin(metaclass=abc.ABCMeta):
    # This mixin provides a method to push to hub without creating a model card
    # and without creating the repository if it does not exist.

    @abc.abstractmethod
    def save_pretrained(
        self,
        save_directory: str,
        *,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None: ...
    def safe_push_to_hub(
        self,
        *,
        repo_id: str,
        commit_message: str,
        token: str,
        create_pr: bool = False,
        safe_serialization: bool = True,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir, safe_serialization=safe_serialization)

            upload_folder(
                repo_id=repo_id,
                folder_path=tmpdir,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )
