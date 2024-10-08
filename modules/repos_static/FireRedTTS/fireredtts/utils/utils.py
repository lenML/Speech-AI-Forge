import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def load_audio(audiopath, sampling_rate):
    """_summary_

    Args:
        audiopath (_type_): audio_path
        sampling_rate (_type_): sampling_rate

    Returns:
        _type_: _description_
    """
    audio, lsr = torchaudio.load(audiopath)

    # stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # resample
    audio_resampled = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")

    if torch.any(audio_resampled > 10) or not torch.any(audio_resampled < 0):
        print(
            f"Error with {audiopath}. Max={audio_resampled.max()} min={audio_resampled.min()}"
        )
    # clip audio invalid values
    audio.clip_(-1, 1)
    audio_resampled.clip_(-1, 1)
    return audio, lsr, audio_resampled
