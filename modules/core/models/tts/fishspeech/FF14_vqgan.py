import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from modules.devices import devices
from modules.repos_static.fish_speech.fish_speech.models.vqgan.modules.firefly import (
    FireflyArchitecture,
)

logger = logging.getLogger(__name__)


class FF14_vqgan:
    """
    封装 fishspeech vqgan
    """

    MODEL_PATH = Path("./models/fish-speech-1_4")

    def __init__(self) -> None:
        self.model: FireflyArchitecture = None
        self.config_name = "firefly_gan_vq"
        self.checkpoint_filename = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.device = devices.get_device_for("fish-speech")
        self.repo_path = Path("./modules/repos_static/fish_speech")

        self.model = self.load_model()

    def unload(self):
        if self.model is None:
            return
        self.model.to("cpu")
        del self.model

    def load_model(self) -> None:
        config_name = self.config_name
        checkpoint_path = self.MODEL_PATH / self.checkpoint_filename
        device = self.device

        hydra.core.global_hydra.GlobalHydra.instance().clear()

        # NOTE: 这个 initialize... 必须是相对路径... 很迷惑
        with initialize(version_base="1.3", config_path="./configs"):
            cfg = compose(config_name=config_name)

        model: FireflyArchitecture = instantiate(cfg)
        state_dict = torch.load(
            checkpoint_path, map_location=device, mmap=True, weights_only=True
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }

        result = model.load_state_dict(state_dict, strict=False, assign=True)
        model.eval()
        model.to(device)

        logger.info(f"Loaded model: {result}")
        return model

    def encode(self, wav: np.ndarray) -> torch.Tensor:
        """
        这里假设 wav 都是 resample 并 mono 过
        """
        ref_audio = torch.from_numpy(wav).clone()
        ref_audio = ref_audio[None].to(self.device)
        # print(ref_audio.shape, wav.shape)
        audio_lengths = torch.tensor(
            [ref_audio.shape[1]], device=self.device, dtype=torch.long
        )
        indices = self.model.encode(ref_audio, audio_lengths)[0][0]

        return indices

    def decode(self, indices: torch.Tensor | np.ndarray) -> np.ndarray:
        model = self.model
        device = self.device

        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices)

        indices = indices.to(device)
        feature_lengths = torch.tensor([indices.shape[1]], device=device)
        fake_audios, _ = model.decode(
            indices=indices[None], feature_lengths=feature_lengths
        )

        fake_audio = fake_audios[0, 0].detach().float().cpu().numpy()
        return fake_audio
