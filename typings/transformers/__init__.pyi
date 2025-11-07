import pathlib
from collections import OrderedDict, UserDict
from typing import Any, Literal

import torch
from PIL import Image

from finegrain_toolbox.types import Self

class BatchEncoding(UserDict[str, torch.Tensor]):
    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BatchEncoding: ...
    @property
    def input_ids(self) -> torch.Tensor: ...

class PreTrainedModel:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
        dtype: torch.dtype,
        subfolder: str = "",
        device_map: dict[str, torch.device] | None = None,
    ) -> Self: ...
    def to(self, device: torch.device) -> Self: ...
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor: ...
    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> torch.nn.Embedding: ...

class Processor:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
    ) -> Self: ...
    def __call__(
        self,
        images: Image.Image | list[Image.Image],
        text: str | list[str] | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding: ...
    def batch_decode(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool,
    ) -> list[str]: ...

class AddedToken:
    def __init__(
        self,
        content: str,
        single_word: bool = False,
        lstrip: bool = False,
        rstrip: bool = False,
        special: bool = False,
        normalized: bool | None = None,
    ) -> None: ...

class PreTrainedTokenizerBase:
    vocab_size: int
    clean_up_tokenization_spaces: bool

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
        subfolder: str = "",
    ) -> Self: ...
    def __call__(
        self,
        text: str | list[str] | None = None,
        padding: bool | str = False,
        max_length: int | None = None,
        truncation: bool | str | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding: ...
    def __len__(self) -> int: ...
    def batch_decode(
        self,
        sequences: torch.Tensor,
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> list[str]: ...
    def add_tokens(
        self,
        new_tokens: str | AddedToken | list[str | AddedToken],
        special_tokens: bool = False,
    ) -> int: ...

class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    @property
    def vocab(self) -> dict[str, int]: ...

class PretrainedConfig:
    pass

class Blip2Config(PretrainedConfig):
    image_token_index: int | None

class Blip2ForConditionalGeneration(PreTrainedModel):
    config: Blip2Config

class Blip2Processor(Processor):
    num_query_tokens: int | None
    tokenizer: T5TokenizerFast

class T5TokenizerFast(PreTrainedTokenizerFast):
    pass

class PreTrainedTokenizer(PreTrainedTokenizerBase):
    pass

class CLIPTokenizer(PreTrainedTokenizer):
    pass

class CLIPPreTrainedModel(PreTrainedModel):
    pass

class ModelOutput(OrderedDict[str, Any]):
    pass

class BaseModelOutput(ModelOutput):
    def __getitem__(self, k: str | int) -> torch.Tensor: ...

class BaseModelOutputWithPooling(ModelOutput):
    pooler_output: torch.Tensor

class CLIPTextModel(CLIPPreTrainedModel):
    def __call__(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithPooling: ...

class T5PreTrainedModel(PreTrainedModel):
    pass

class T5EncoderModel(T5PreTrainedModel):
    def __call__(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput: ...
