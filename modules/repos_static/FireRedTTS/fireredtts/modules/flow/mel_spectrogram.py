from functools import partial
import torch
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
from torchaudio.functional import resample as ta_resample_fn

MAX_WAV_VALUE = 32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        str_key_mel_basis = str(fmax) + "_" + str(y.device)
        mel_basis[str_key_mel_basis] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str_key_mel_basis], spec)
    spec = spectral_normalize_torch(spec)

    return spec


kaiser_best_resampling_fn = partial(
    ta_resample_fn, 
    resampling_method="sinc_interp_kaiser", # DO NOT CHANGE!
    rolloff=0.917347, # DO NOT CHANGE!
    beta=12.9846, # DO NOT CHANGE!
    lowpass_filter_width=50, # DO NOT CHANGE!
)


class MelSpectrogramExtractor(object):
    def __init__(
        self,
        n_fft=1024,
        win_size=1024,
        num_mels=100,
        hop_size=160,
        sampling_rate=16000,
        fmin=0,
        fmax=None,
    ):
        self.n_fft = n_fft
        self.win_size = win_size
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, wav_path) -> np.ndarray:
        wav_data, wav_sr = librosa.load(wav_path, sr=None, mono=True)
        wav_data = torch.from_numpy(wav_data.copy()).unsqueeze(0)
        # for 16k wavs, up-downsample to reduce artifects
        if wav_sr == self.sampling_rate:
            wav_data = kaiser_best_resampling_fn(wav_data, orig_freq=wav_sr, new_freq=24000)
            wav_data = kaiser_best_resampling_fn(wav_data, orig_freq=24000, new_freq=self.sampling_rate)
        else:
            wav_data = kaiser_best_resampling_fn(wav_data, orig_freq=wav_sr, new_freq=self.sampling_rate)        

        # (1, num_mels, t)
        mel = mel_spectrogram(
            wav_data,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax,
        )
        mel = mel.squeeze(0).transpose(1, 0)
        return mel  # (t, num_mels)
