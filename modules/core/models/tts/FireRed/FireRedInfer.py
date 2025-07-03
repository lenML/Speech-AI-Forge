import gc
import json
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torchaudio

from modules.core.models.tts.FireRed.FRBepTokenizer import FRBepTokenizer
from modules.core.models.tts.FireRed.MelExtractor import MelSpectrogramExtractor
from modules.repos_static.FireRedTTS.fireredtts.modules import Token2Wav
from modules.repos_static.FireRedTTS.fireredtts.modules.codec.speaker import (
    SpeakerEmbedddingExtractor,
)
from modules.repos_static.FireRedTTS.fireredtts.modules.gpt.gpt import GPT
from modules.repos_static.FireRedTTS.fireredtts.utils.utils import load_audio


@dataclass(frozen=True, repr=False, eq=False)
class FireRedTTSParams:
    top_p: float = 0.85
    top_k: float = 30
    temperature: float = 0.75
    num_return_sequences: int = 9
    num_beams: int = 1
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0


def_ps = FireRedTTSParams()


class FireRedTTSInfer:

    def __init__(
        self,
        config_path: str,
        pretrained_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.config: dict = json.load(open(config_path))
        self.gpt_path: str = os.path.join(pretrained_path, "fireredtts_gpt.pt")
        self.token2wav_path: str = os.path.join(
            pretrained_path, "fireredtts_token2wav.pt"
        )
        self.speaker_extractor_path: str = os.path.join(
            pretrained_path, "fireredtts_speaker.bin"
        )

        assert os.path.exists(self.token2wav_path)
        assert os.path.exists(self.gpt_path)
        assert os.path.exists(self.speaker_extractor_path)

        # Tokenizer
        self.text_tokenizer: FRBepTokenizer = FRBepTokenizer()

        # Speaker extractor
        self.speaker_extractor: SpeakerEmbedddingExtractor = SpeakerEmbedddingExtractor(
            ckpt_path=self.speaker_extractor_path, device=device
        )

        # Load GPT model
        self.gpt: GPT = GPT(
            start_text_token=self.config["gpt"]["gpt_start_text_token"],
            stop_text_token=self.config["gpt"]["gpt_stop_text_token"],
            layers=self.config["gpt"]["gpt_layers"],
            model_dim=self.config["gpt"]["gpt_n_model_channels"],
            heads=self.config["gpt"]["gpt_n_heads"],
            max_text_tokens=self.config["gpt"]["gpt_max_text_tokens"],
            max_mel_tokens=self.config["gpt"]["gpt_max_audio_tokens"],
            max_prompt_tokens=self.config["gpt"]["gpt_max_prompt_tokens"],
            code_stride_len=self.config["gpt"]["gpt_code_stride_len"],
            number_text_tokens=self.config["gpt"]["gpt_number_text_tokens"],
            num_audio_tokens=self.config["gpt"]["gpt_num_audio_tokens"],
            start_audio_token=self.config["gpt"]["gpt_start_audio_token"],
            stop_audio_token=self.config["gpt"]["gpt_stop_audio_token"],
        )

        sd = torch.load(self.gpt_path, map_location="cpu")["model"]
        self.gpt.load_state_dict(sd, strict=True)
        self.gpt = self.gpt.to(device=device, dtype=dtype)
        self.gpt.eval()
        self.gpt.init_gpt_for_inference(kv_cache=True)

        # Mel-spectrogram extractor
        self.mel_extractor: MelSpectrogramExtractor = MelSpectrogramExtractor()

        # Load token2wav model
        self.token2wav: Token2Wav = Token2Wav.init_from_config(self.config)
        sd = torch.load(self.token2wav_path, map_location="cpu")
        self.token2wav.load_state_dict(sd, strict=True)
        self.token2wav.generator.remove_weight_norm()
        self.token2wav.eval()
        self.token2wav = self.token2wav.to(device=device, dtype=dtype)

    def unload_models(self):
        """卸载所有模型并释放资源。"""
        if self.gpt is None:
            return
        del self.gpt
        del self.token2wav
        del self.speaker_extractor
        self.gpt = None
        self.token2wav = None
        self.speaker_extractor = None

    def extract_spk_embeddings(self, audio: np.ndarray, audio_sr: int) -> torch.Tensor:
        """Extract speaker embeddings from numpy audio array.

        Args:
            audio (np.ndarray): Input audio as a float32 numpy array.
            audio_sr (int): Sampling rate of the input audio.

        Returns:
            torch.Tensor: Speaker embeddings tensor.
        """
        # Convert numpy array to torch tensor and resample if needed
        audio_tensor = (
            torch.from_numpy(audio)
            .unsqueeze(0)
            .float()
            .to(device=self.device, dtype=self.dtype)
        )
        audio_resampled = torchaudio.functional.resample(
            audio_tensor, orig_freq=audio_sr, new_freq=16000
        )
        audio_len = torch.tensor(
            [audio_resampled.shape[1]], dtype=torch.long, requires_grad=False
        )

        # Extract speaker embeddings
        spk_embeddings: torch.Tensor = (
            self.speaker_extractor(audio_resampled)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        return spk_embeddings

    def do_gpt_inference(
        self,
        spk_gpt: torch.Tensor,
        text_tokens: torch.Tensor,
        params: FireRedTTSParams = FireRedTTSParams(),
    ) -> torch.Tensor:
        """Perform GPT inference to generate audio codes.

        Args:
            spk_gpt (torch.Tensor): Speaker embedding for GPT.
            text_tokens (torch.Tensor): Tokenized input text.

        Returns:
            torch.Tensor: Generated GPT codes.
        """
        with torch.no_grad():
            gpt_codes = self.gpt.generate(
                cond_latents=spk_gpt,
                text_inputs=text_tokens,
                input_tokens=None,
                do_sample=True,
                top_p=params.top_p or def_ps.top_p,
                top_k=params.top_k or def_ps.top_k,
                temperature=params.temperature or def_ps.temperature,
                num_return_sequences=params.num_return_sequences
                or def_ps.num_return_sequences,
                num_beams=params.num_beams or def_ps.num_beams,
                length_penalty=params.length_penalty or def_ps.length_penalty,
                repetition_penalty=params.repetition_penalty
                or def_ps.repetition_penalty,
                output_attentions=False,
            )

        EOS_TOKEN = self.config["gpt"]["gpt_stop_audio_token"]
        seqs = [
            seq[: (seq == EOS_TOKEN).nonzero(as_tuple=True)[0][0]] for seq in gpt_codes
        ]
        sorted_seqs = sorted(seqs, key=lambda i: len(i))
        gpt_codes = sorted_seqs[2].unsqueeze(0)

        return gpt_codes

    def synthesize(
        self,
        audio: np.ndarray,
        audio_sr: int,
        text: str,
        lang: str = "auto",
        params: FireRedTTSParams = FireRedTTSParams(),
    ) -> torch.Tensor:
        """Synthesize speech from text and speaker audio.

        Args:
            audio (np.ndarray): Input audio as a float32 numpy array.
            audio_sr (int): Sampling rate of the input audio.
            text (str): Input text.
            lang (str, optional): Language of the text ('zh', 'en', or 'auto'). Defaults to "auto".

        Returns:
            torch.Tensor: Synthesized audio waveform.
        """
        # Only supports Chinese and English
        assert lang in ["zh", "en", "auto"]
        # audio is Float32 array
        assert audio.dtype == np.float32

        # Convert text to tokens
        text_tokens: torch.Tensor = (
            torch.IntTensor(self.text_tokenizer.encode(text=text, lang=lang))
            .unsqueeze(0)
            .to(self.device)
        )
        assert text_tokens.shape[-1] < 400

        # Extract speaker embedding
        spk_embeddings = self.extract_spk_embeddings(
            audio=audio, audio_sr=audio_sr
        ).unsqueeze(0)
        with torch.no_grad():
            spk_gpt = self.gpt.reference_embedding(spk_embeddings)

        # Perform GPT inference
        gpt_start_time = time.time()
        gpt_codes = self.do_gpt_inference(
            spk_gpt=spk_gpt, text_tokens=text_tokens, params=params
        )
        gpt_end_time = time.time()

        # Extract mel-spectrogram from input audio
        prompt_mel = (
            self.mel_extractor.from_array(wav_data=audio, wav_sr=audio_sr)
            .unsqueeze(0)
            .to(self.device)
        )

        # Convert tokens to waveform
        voc_start_time = time.time()
        rec_wavs = self.token2wav.inference(gpt_codes, prompt_mel, n_timesteps=10)
        voc_end_time = time.time()

        return rec_wavs
