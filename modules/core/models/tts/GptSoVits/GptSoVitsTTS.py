import torch
from modules.repos_static.GPT_SoVITS.GPT_SoVITS.TTS_infer_pack.TTS import (
    TTS,
)
from modules.repos_static.GPT_SoVITS.GPT_SoVITS.BigVGAN.bigvgan import BigVGAN
from modules.repos_static.GPT_SoVITS.GPT_SoVITS.module.models import Generator


# 主要修改: 加载路径
class GptSoVitsTTS(TTS):
    def init_vocoder(self, version: str):
        if version == "v3":
            if (
                self.vocoder is not None
                and self.vocoder.__class__.__name__ == "BigVGAN"
            ):
                return
            if self.vocoder is not None:
                self.vocoder.cpu()
                del self.vocoder
                self.empty_cache()

            self.vocoder = BigVGAN.from_pretrained(
                "models/gpt_sovits_v3/models--nvidia--bigvgan_v2_24khz_100band_256x",
                use_cuda_kernel=False,
            )  # if True, RuntimeError: Ninja is required to load C++ extensions
            # remove weight norm in the model and set to eval mode
            self.vocoder.remove_weight_norm()

            self.vocoder_configs["sr"] = 24000
            self.vocoder_configs["T_ref"] = 468
            self.vocoder_configs["T_chunk"] = 934
            self.vocoder_configs["upsample_rate"] = 256
            self.vocoder_configs["overlapped_len"] = 12

        elif version == "v4":
            if (
                self.vocoder is not None
                and self.vocoder.__class__.__name__ == "Generator"
            ):
                return
            if self.vocoder is not None:
                self.vocoder.cpu()
                del self.vocoder
                self.empty_cache()

            self.vocoder = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            self.vocoder.remove_weight_norm()
            state_dict_g = torch.load(
                "models/gpt_sovits_v4/gsv-v4-pretrained/vocoder.pth",
                map_location="cpu",
            )
            print("loading vocoder", self.vocoder.load_state_dict(state_dict_g))

            self.vocoder_configs["sr"] = 48000
            self.vocoder_configs["T_ref"] = 500
            self.vocoder_configs["T_chunk"] = 1000
            self.vocoder_configs["upsample_rate"] = 480
            self.vocoder_configs["overlapped_len"] = 12

        self.vocoder = self.vocoder.eval()
        if self.configs.is_half == True:
            self.vocoder = self.vocoder.half().to(self.configs.device)
        else:
            self.vocoder = self.vocoder.to(self.configs.device)
