import os
from typing import Union

import soundfile as sf
import torch
import torchaudio

from modules.Denoiser.AudioDenoiser import AudioDenoiser
from modules.devices import devices
from modules.utils.constants import MODELS_DIR

ad: Union[AudioDenoiser, None] = None


class TTSAudioDenoiser:

    def load_ad(self):
        global ad
        if ad is None:
            ad = AudioDenoiser(
                os.path.join(
                    MODELS_DIR,
                    "Denoise",
                    "audio-denoiser-512-32-v1",
                ),
                device=devices.device,
            )
            ad.model.to(devices.device)
        return ad

    def denoise(self, audio_data, sample_rate, auto_scale=False):
        ad = self.load_ad()
        sr = ad.model_sample_rate
        return sr, ad.process_waveform(audio_data, sample_rate, auto_scale)


if __name__ == "__main__":
    tts_deno = TTSAudioDenoiser()
    data, sr = sf.read("test.wav")
    audio_tensor = torch.from_numpy(data).unsqueeze(0).float()
    print(audio_tensor)

    # data, sr = torchaudio.load("test.wav")
    # print(data)
    # data = data.to(devices.device)

    sr, denoised = tts_deno.denoise(audio_data=audio_tensor, sample_rate=sr)
    denoised = denoised.cpu()
    torchaudio.save("denoised.wav", denoised, sample_rate=sr)
