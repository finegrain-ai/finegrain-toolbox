import dataclasses as dc
import pathlib

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from ..dc import DcMixin
from ..torch import default_device, default_dtype
from ..types import Self


@dc.dataclass(kw_only=True)
class Tokenized:
    tokens: torch.Tensor
    text: str


@dc.dataclass(kw_only=True)
class Prompt(DcMixin):
    embeds: torch.Tensor  # (batch_size, 768)
    text: str
    batch_size: int = 1

    def repeat(self, batch_size: int) -> Self:
        assert self.batch_size == 1
        return self.__class__(
            embeds=torch.cat([self.embeds for _ in range(batch_size)]),
            text=self.text,
            batch_size=batch_size,
        )

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        params = self.shallow_asdict()
        params["embeds"] = self.embeds.to(device=device, dtype=dtype)
        return self.__class__(**params)


@dc.dataclass(kw_only=True)
class TextEncoder:
    device: torch.device
    dtype: torch.dtype
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    max_sequence_length: int = 77

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str | pathlib.Path,
        device: torch.device = default_device,
        dtype: torch.dtype = default_dtype,
    ) -> Self:
        tokenizer = CLIPTokenizer.from_pretrained(path_or_id, subfolder="tokenizer")

        text_encoder = CLIPTextModel.from_pretrained(
            path_or_id,
            subfolder="text_encoder",
            dtype=dtype,
            device_map={"": device},
        )

        return cls(
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    @torch.inference_mode()
    def tokenize(self, text: str) -> Tokenized:
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        return Tokenized(text=text, tokens=tokens)

    @torch.inference_mode()
    def encode_tokens(self, tokens: Tokenized) -> Prompt:
        prompt_embeds = self.text_encoder(
            tokens.tokens,
            output_hidden_states=False,
        ).last_hidden_state.to(dtype=self.dtype, device=self.device)
        return Prompt(embeds=prompt_embeds, text=tokens.text)

    def encode(self, text: str) -> Prompt:
        return self.encode_tokens(self.tokenize(text))
