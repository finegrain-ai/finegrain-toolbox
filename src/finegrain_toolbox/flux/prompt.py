import dataclasses as dc
import pathlib

import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from ..torch import default_device, default_dtype
from ..types import Self


@dc.dataclass(kw_only=True)
class Tokenized:
    clip_tokens: torch.Tensor
    t5_tokens: torch.Tensor
    text: str


@dc.dataclass(kw_only=True)
class Prompt:
    t5_embeds: torch.Tensor  # (batch_size, seq_len, 4096)
    clip_embeds: torch.Tensor  # (batch_size, 768)
    text_ids: torch.Tensor  # shape (seq_len, 3)
    text: str
    batch_size: int

    def repeat(self, batch_size: int) -> Self:
        assert self.batch_size == 1
        return self.__class__(
            t5_embeds=torch.cat([self.t5_embeds for _ in range(batch_size)]),
            clip_embeds=torch.cat([self.clip_embeds for _ in range(batch_size)]),
            text_ids=self.text_ids,
            text=self.text,
            batch_size=batch_size,
        )

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        params = dc.asdict(self)
        params["t5_embeds"] = self.t5_embeds.to(device=device, dtype=dtype)
        params["clip_embeds"] = self.clip_embeds.to(device=device, dtype=dtype)
        params["text_ids"] = self.text_ids.to(device=device, dtype=dtype)
        return self.__class__(**params)


def prompt_with_embeds(
    text: str,
    clip_prompt_embeds: torch.Tensor,
    t5_prompt_embeds: torch.Tensor,
) -> Prompt:
    device, dtype = clip_prompt_embeds.device, clip_prompt_embeds.dtype
    assert t5_prompt_embeds.device == device and t5_prompt_embeds.dtype == dtype

    text_ids = torch.zeros(t5_prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return Prompt(
        t5_embeds=t5_prompt_embeds,
        clip_embeds=clip_prompt_embeds,
        text_ids=text_ids,
        text=text,
        batch_size=1,
    )


@dc.dataclass(kw_only=True)
class TextEncoder:
    device: torch.device
    dtype: torch.dtype
    clip_tokenizer: CLIPTokenizer
    t5_tokenizer: T5TokenizerFast
    clip_text_encoder: CLIPTextModel
    t5_text_encoder: T5EncoderModel
    clip_max_sequence_length: int = 77
    t5_max_sequence_length: int = 512

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str | pathlib.Path,
        device: torch.device = default_device,
        dtype: torch.dtype = default_dtype,
    ) -> Self:
        clip_tokenizer = CLIPTokenizer.from_pretrained(
            path_or_id,
            subfolder="tokenizer",
        )

        t5_tokenizer = T5TokenizerFast.from_pretrained(
            path_or_id,
            subfolder="tokenizer_2",
        )

        clip_text_encoder = CLIPTextModel.from_pretrained(
            path_or_id,
            subfolder="text_encoder",
            torch_dtype=dtype,
            device_map={"": device},
        )
        t5_text_encoder = T5EncoderModel.from_pretrained(
            path_or_id,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            device_map={"": device},
        )

        return cls(
            device=device,
            dtype=dtype,
            clip_tokenizer=clip_tokenizer,
            t5_tokenizer=t5_tokenizer,
            clip_text_encoder=clip_text_encoder,
            t5_text_encoder=t5_text_encoder,
        )

    @torch.inference_mode()
    def tokenize(self, text: str) -> Tokenized:
        clip_tokens = self.clip_tokenizer(
            text,
            padding="max_length",
            max_length=self.clip_max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        t5_tokens = self.t5_tokenizer(
            text,
            padding="max_length",
            max_length=self.t5_max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        return Tokenized(text=text, clip_tokens=clip_tokens, t5_tokens=t5_tokens)

    @torch.inference_mode()
    def encode_tokens(self, tokens: Tokenized) -> Prompt:
        clip_prompt_embeds = self.clip_text_encoder(
            tokens.clip_tokens,
            output_hidden_states=False,
        ).pooler_output.to(dtype=self.dtype, device=self.device)

        t5_prompt_embeds = self.t5_text_encoder(
            tokens.t5_tokens,
            output_hidden_states=False,
        )[0].to(dtype=self.dtype, device=self.device)

        return prompt_with_embeds(tokens.text, clip_prompt_embeds, t5_prompt_embeds)

    def encode(self, text: str) -> Prompt:
        return self.encode_tokens(self.tokenize(text))
