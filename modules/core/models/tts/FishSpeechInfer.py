import io
import logging
from typing import Optional, Union

import librosa
import numpy as np
import torch
from transformers import LlamaTokenizer

from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.devices import devices
from modules.repos_static.fish_speech.fish_speech.conversation import (
    CODEBOOK_PAD_TOKEN_ID,
)
from modules.repos_static.fish_speech.fish_speech.models.text2semantic.llama import (
    DualARTransformer,
    NaiveTransformer,
)
from modules.repos_static.fish_speech.fish_speech.models.vqgan.modules.firefly import (
    FireflyArchitecture,
)
from modules.repos_static.fish_speech.fish_speech.text.clean import clean_text
from modules.repos_static.fish_speech.tools.llama.generate import (
    decode_n_tokens,
    decode_one_token_ar,
    decode_one_token_naive,
)

FISH_SPEECH_LLAMA = Union[NaiveTransformer, DualARTransformer]


class FishSpeechInfer:
    model_id = "fish_speech"

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        llama: FISH_SPEECH_LLAMA,
        token_decoder: callable,
        vqgan: FireflyArchitecture,
    ) -> None:
        self.llama = llama
        self.token_decoder = token_decoder
        self.vqgan = vqgan

        self.device = devices.get_device_for(self.model_id)
        self.dtype = devices.dtype
        self.tokenizer: LlamaTokenizer = self.llama.tokenizer
        self.im_end_id: int = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @property
    def sample_rate(self) -> int:
        return self.vqgan.spec_transform.sample_rate

    @torch.no_grad()
    @torch.inference_mode()
    def infer(
        self,
        *,
        text: str = "",
        spk: Optional[TTSSpeaker] = None,
        emotion: Optional[str] = None,
        max_new_tokens: int = 1024,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
        # NOTE: 之前的生成结果，添加在 encoded prompt 后面
        encoded_prefix: list[torch.Tensor] = [],
    ):
        encoded_text = self.encode_tokens(text=text)
        encoded_prompt = self.encode_spk(spk=spk, emotion=emotion) if spk else None

        global_encoded = encoded_prefix + [encoded_text]
        partial_encoded = (
            [encoded_prompt] + global_encoded if encoded_prompt else global_encoded
        )

        # Move temperature, top_p, repetition_penalty to device
        # This is important so that changing params doesn't trigger recompile
        # ref: https://github.com/fishaudio/fish-speech/blob/cee143d213906ccaf0b81a86833bc5f0289d4c5d/tools/llama/generate.py#L434-L440
        temperature = torch.tensor(temperature, device=self.device, dtype=torch.float)
        top_p = torch.tensor(top_p, device=self.device, dtype=torch.float)
        repetition_penalty = torch.tensor(
            repetition_penalty, device=self.device, dtype=torch.float
        )

        cat_encoded = torch.cat(partial_encoded, dim=1)
        prompt_length = cat_encoded.size(1)

        y: torch.Tensor = self.generate(
            prompt=cat_encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        # Put the generated tokens
        # since there is <im_end> and <eos> tokens, we remove last 2 tokens
        codes = y[1:, prompt_length:-1].clone()
        codes = codes - 1
        assert (codes >= 0).all(), f"Negative code found"

        decoded = y[:, prompt_length:-1].clone()

        generated = self.decode_vq_tokens(codes=codes)

        return decoded, generated

    @torch.no_grad()
    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: torch.Tensor,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """`
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """

        model = self.llama
        im_end_id = self.im_end_id
        decode_one_token = self.token_decoder

        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        # create an empty tensor of the expected final shape and fill in the current tokens

        T = prompt.size(1)

        if max_new_tokens:
            if T + max_new_tokens > model.config.max_seq_len:
                max_new_tokens = model.config.max_seq_len - T
                self.logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

            T_new = T + max_new_tokens
        else:
            T_new = model.config.max_seq_len
            max_new_tokens = T_new - T

        device, dtype = prompt.device, prompt.dtype
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=T_new,
                dtype=next(model.parameters()).dtype,
            )

        codebook_dim = 1 + model.config.num_codebooks
        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty((codebook_dim, T_new), dtype=dtype, device=device)
        empty[:, :T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        # Use non-accelerated version for now, to avoid compilation overhead
        prefill_decode = (
            decode_one_token_naive
            if isinstance(model, NaiveTransformer)
            else decode_one_token_ar
        )

        next_token = prefill_decode(
            model, prompt.view(1, codebook_dim, -1), input_pos, **sampling_kwargs
        )
        seq[:, T : T + 1] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        x = decode_n_tokens(
            model,
            next_token.view(1, codebook_dim, -1),
            input_pos,
            max_new_tokens - 1,
            im_end_id=im_end_id,
            decode_one_token=decode_one_token,
            **sampling_kwargs,
        )
        # x = torch.cat(generated_tokens, dim=1)
        seq = seq[:, : T + 1 + x.size(1)]
        seq[:, T + 1 :] = x

        return seq

    def cuda_synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # TODO: 支持 多 shot prompt
    def encode_spk(self, spk: TTSSpeaker, emotion: Optional[str] = None):
        ref_config = spk.get_ref(lambda ref: ref.emotion == emotion)
        if ref_config is None:
            return None

        wav_bytes = ref_config.wav
        wav_sr = ref_config.wav_sr
        prompt_text = ref_config.text

        prompt_tokens = (
            self.encode_ref_wav(wav_bytes=wav_bytes, wav_sr=wav_sr)
            if wav_bytes is not None
            else None
        )

        encoded_prompt = self.encode_tokens(
            text=prompt_text, prompt_tokens=prompt_tokens
        )

        return encoded_prompt

    def encode_ref_wav(self, *, wav_bytes: bytes, sr: int):
        audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=sr, mono=True)
        audios = torch.from_numpy(audio).to(device=self.device)[None, None, :]
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=self.device, dtype=torch.long
        )
        prompt_tokens = self.vqgan.encode(audios, audio_lengths)[0][0]

        return prompt_tokens

    def encode_tokens(self, *, text: str, prompt_tokens: Optional[torch.Tensor] = None):
        model = self.llama
        tokenizer = self.tokenizer
        num_codebooks = model.config.num_codebooks

        text = clean_text(text)
        text = f"<|im_start|>user\n{text}<|im_end|><|im_start|>assistant\n"

        new_tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=10**6,
            truncation=False,
        )
        tokens = torch.tensor([new_tokens], dtype=torch.int, device=self.device)

        # Codebooks
        zeros = (
            torch.ones(
                (num_codebooks, tokens.size(1)), dtype=torch.int, device=self.device
            )
            * CODEBOOK_PAD_TOKEN_ID
        )
        prompt = torch.cat((tokens, zeros), dim=0)

        if prompt_tokens is None:
            return prompt

        # Get prompt tokens
        if prompt_tokens.ndim == 3:
            assert (
                prompt_tokens.shape[0] == 1
            ), f"3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
            prompt_tokens = prompt_tokens[0]

        assert prompt_tokens.ndim == 2
        data = prompt_tokens + 1

        if prompt_tokens.shape[0] > num_codebooks:
            self.logger.warning(
                f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
            )
            data = data[:num_codebooks]

        # Add pad token for each codebook
        data = torch.cat(
            (data, torch.zeros((data.size(0), 1), dtype=torch.int, device=self.device)),
            dim=1,
        )

        # Since 1.0, we use <|semantic|>
        s0_token_id = tokenizer.convert_tokens_to_ids("<|semantic|>")
        end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        main_token_ids = (
            torch.ones((1, data.size(1)), dtype=torch.int, device=self.device)
            * s0_token_id
        )
        main_token_ids[0, -1] = end_token_id

        data = torch.cat((main_token_ids, data), dim=0)
        prompt = torch.cat((prompt, data), dim=1)

        return prompt

    def decode_vq_tokens(self, *, codes: torch.Tensor) -> np.ndarray:
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            feature_lengths = torch.tensor([codes.shape[1]], device=self.device)
            fake_audios = self.vqgan.decode(
                indices=codes[None], feature_lengths=feature_lengths
            ).squeeze()

        fake_audios: np.ndarray = fake_audios.float().cpu().numpy()
        return fake_audios
