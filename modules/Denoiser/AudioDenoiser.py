import logging
import math
from typing import Union

import torch
import torchaudio
from audio_denoiser.helpers.audio_helper import (
    create_spectrogram,
    reconstruct_from_spectrogram,
)
from audio_denoiser.helpers.torch_helper import batched_apply
from torch import nn

from modules.Denoiser.AudioNosiseModel import load_audio_denosier_model

_expected_t_std = 0.23
_recommended_backend = "soundfile"


# ref: https://github.com/jose-solorzano/audio-denoiser
class AudioDenoiser:
    def __init__(
        self,
        local_dir: str,
        device: Union[str, torch.device] = None,
        num_iterations: int = 100,
    ):
        super().__init__()
        if device is None:
            is_cuda = torch.cuda.is_available()
            if not is_cuda:
                logging.warning("CUDA not available. Will use CPU.")
            device = torch.device("cuda:0") if is_cuda else torch.device("cpu")
        self.device = device
        self.model = load_audio_denosier_model(dir_path=local_dir, device=device)
        self.model.eval()
        self.model_sample_rate = self.model.sample_rate
        self.scaler = self.model.scaler
        self.n_fft = self.model.n_fft
        self.segment_num_frames = self.model.num_frames
        self.num_iterations = num_iterations

    @staticmethod
    def _sp_log(spectrogram: torch.Tensor, eps=0.01):
        return torch.log(spectrogram + eps)

    @staticmethod
    def _sp_exp(log_spectrogram: torch.Tensor, eps=0.01):
        return torch.clamp(torch.exp(log_spectrogram) - eps, min=0)

    @staticmethod
    def _trimmed_dev(waveform: torch.Tensor, q: float = 0.90) -> float:
        # Expected for training data is ~0.23
        abs_waveform = torch.abs(waveform)
        quantile_value = torch.quantile(abs_waveform, q).item()
        trimmed_values = waveform[abs_waveform >= quantile_value]
        return torch.std(trimmed_values).item()

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        return_cpu_tensor: bool = False,
        auto_scale: bool = False,
    ) -> torch.Tensor:
        """
        Denoises a waveform.
        @param waveform: A waveform tensor. Use torchaudio structure.
        @param sample_rate: The sample rate of the waveform in Hz.
        @param return_cpu_tensor: Whether the returned tensor must be a CPU tensor.
        @param auto_scale: Normalize the scale of the waveform before processing. Recommended for low-volume audio.
        @return: A denoised waveform.
        """
        waveform = waveform.cpu()
        if auto_scale:
            w_t_std = self._trimmed_dev(waveform)
            waveform = waveform * _expected_t_std / w_t_std
        if sample_rate != self.model_sample_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.model_sample_rate
            )
            waveform = transform(waveform)
        hop_len = self.n_fft // 2
        spectrogram = create_spectrogram(waveform, n_fft=self.n_fft, hop_length=hop_len)
        spectrogram = spectrogram.to(self.device)
        num_a_channels = spectrogram.size(0)
        with torch.no_grad():
            results = []
            for c in range(num_a_channels):
                c_spectrogram = spectrogram[c]
                # c_spectrogram: (257, num_frames)
                fft_size, num_frames = c_spectrogram.shape
                num_segments = math.ceil(num_frames / self.segment_num_frames)
                adj_num_frames = num_segments * self.segment_num_frames
                if adj_num_frames > num_frames:
                    c_spectrogram = nn.functional.pad(
                        c_spectrogram, (0, adj_num_frames - num_frames)
                    )
                c_spectrogram = c_spectrogram.view(
                    fft_size, num_segments, self.segment_num_frames
                )
                # c_spectrogram: (257, num_segments, 32)
                c_spectrogram = torch.permute(c_spectrogram, (1, 0, 2))
                # c_spectrogram: (num_segments, 257, 32)
                log_c_spectrogram = self._sp_log(c_spectrogram)
                scaled_log_c_sp = self.scaler(log_c_spectrogram)
                pred_noise_log_sp = batched_apply(
                    self.model, scaled_log_c_sp, detached=True
                )
                log_denoised_sp = log_c_spectrogram - pred_noise_log_sp
                denoised_sp = self._sp_exp(log_denoised_sp)
                # denoised_sp: (num_segments, 257, 32)
                denoised_sp = torch.permute(denoised_sp, (1, 0, 2))
                # denoised_sp: (257, num_segments, 32)
                denoised_sp = denoised_sp.contiguous().view(1, fft_size, adj_num_frames)
                # denoised_sp: (1, 257, adj_num_frames)
                denoised_sp = denoised_sp[:, :, :num_frames]
                denoised_sp = denoised_sp.cpu()
                denoised_waveform = reconstruct_from_spectrogram(
                    denoised_sp, num_iterations=self.num_iterations
                )
                # denoised_waveform: (1, num_samples)
                results.append(denoised_waveform)
            cpu_results = torch.cat(results)
            return cpu_results if return_cpu_tensor else cpu_results.to(self.device)

    def process_audio_file(
        self, in_audio_file: str, out_audio_file: str, auto_scale: bool = False
    ):
        """
        Denoises an audio file.
        @param in_audio_file: An input audio file with a format supported by torchaudio.
        @param out_audio_file: Am output audio file with a format supported by torchaudio.
        @param auto_scale: Whether the input waveform scale should be normalized before processing. Recommended for low-volume audio.
        """
        waveform, sample_rate = torchaudio.load(in_audio_file)
        denoised_waveform = self.process_waveform(
            waveform, sample_rate, return_cpu_tensor=True, auto_scale=auto_scale
        )
        torchaudio.save(
            out_audio_file, denoised_waveform, sample_rate=self.model_sample_rate
        )
