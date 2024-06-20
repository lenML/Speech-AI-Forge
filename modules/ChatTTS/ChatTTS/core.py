import logging
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from vocos import Vocos

from .infer.api import infer_code, refine_text
from .model.dvae import DVAE
from .model.gpt import GPT_warpper
from .utils.infer_utils import (
    apply_character_map,
    apply_half2full_map,
    count_invalid_characters,
    detect_language,
)
from .utils.io_utils import get_latest_modified_file

logging.basicConfig(level=logging.INFO)


class Chat:
    def __init__(
        self,
    ):
        self.pretrain_models = {}
        self.normalizer = {}
        self.logger = logging.getLogger(__name__)

    def check_model(self, level=logging.INFO, use_decoder=False):
        not_finish = False
        check_list = ["vocos", "gpt", "tokenizer"]

        if use_decoder:
            check_list.append("decoder")
        else:
            check_list.append("dvae")

        for module in check_list:
            if module not in self.pretrain_models:
                self.logger.log(logging.WARNING, f"{module} not initialized.")
                not_finish = True

        if not not_finish:
            self.logger.log(level, f"All initialized.")

        return not not_finish

    def load_models(
        self,
        source="huggingface",
        force_redownload=False,
        local_path="<LOCAL_PATH>",
        **kwargs,
    ):
        if source == "huggingface":
            hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            try:
                download_path = get_latest_modified_file(
                    os.path.join(hf_home, "hub/models--2Noise--ChatTTS/snapshots")
                )
            except:
                download_path = None
            if download_path is None or force_redownload:
                self.logger.log(
                    logging.INFO,
                    f"Download from HF: https://huggingface.co/2Noise/ChatTTS",
                )
                download_path = snapshot_download(
                    repo_id="2Noise/ChatTTS", allow_patterns=["*.pt", "*.yaml"]
                )
            else:
                self.logger.log(logging.INFO, f"Load from cache: {download_path}")
        elif source == "local":
            self.logger.log(logging.INFO, f"Load from local: {local_path}")
            download_path = local_path

        self._load(
            **{
                k: os.path.join(download_path, v)
                for k, v in OmegaConf.load(
                    os.path.join(download_path, "config", "path.yaml")
                ).items()
            },
            **kwargs,
        )

    def _load(
        self,
        vocos_config_path: str = None,
        vocos_ckpt_path: str = None,
        dvae_config_path: str = None,
        dvae_ckpt_path: str = None,
        gpt_config_path: str = None,
        gpt_ckpt_path: str = None,
        decoder_config_path: str = None,
        decoder_ckpt_path: str = None,
        tokenizer_path: str = None,
        device: str = None,
        compile: bool = True,
        dtype: torch.dtype = torch.float32,
        dtype_vocos: torch.dtype = None,
        dtype_dvae: torch.dtype = None,
        dtype_gpt: torch.dtype = None,
        dtype_decoder: torch.dtype = None,
    ):
        assert device is not None, "device should not be None"

        dtype_vocos = dtype_vocos or dtype
        dtype_dvae = dtype_dvae or dtype
        dtype_gpt = dtype_gpt or dtype
        dtype_decoder = dtype_decoder or dtype

        map_location = torch.device("cpu")

        if vocos_config_path:
            vocos = (
                Vocos.from_hparams(vocos_config_path)
                .to(device=device, dtype=dtype_vocos)
                .eval()
            )
            assert vocos_ckpt_path, "vocos_ckpt_path should not be None"
            vocos.load_state_dict(
                torch.load(vocos_ckpt_path, map_location=map_location)
            )
            self.pretrain_models["vocos"] = vocos
            self.logger.log(logging.INFO, "vocos loaded.")

        if dvae_config_path:
            cfg = OmegaConf.load(dvae_config_path)
            dvae = DVAE(**cfg).to(device=device, dtype=dtype_dvae).eval()
            assert dvae_ckpt_path, "dvae_ckpt_path should not be None"
            dvae.load_state_dict(torch.load(dvae_ckpt_path, map_location=map_location))
            self.pretrain_models["dvae"] = dvae
            self.logger.log(logging.INFO, "dvae loaded.")

        if gpt_config_path:
            cfg = OmegaConf.load(gpt_config_path)
            gpt = GPT_warpper(**cfg).to(device=device, dtype=dtype_gpt).eval()
            assert gpt_ckpt_path, "gpt_ckpt_path should not be None"
            gpt.load_state_dict(torch.load(gpt_ckpt_path, map_location=map_location))
            if compile and "cuda" in str(device):
                self.logger.info("compile gpt model")
                try:
                    gpt.gpt.forward = torch.compile(
                        gpt.gpt.forward, backend="inductor", dynamic=True
                    )
                except RuntimeError as e:
                    logging.warning(f"Compile failed,{e}. fallback to normal mode.")
            self.pretrain_models["gpt"] = gpt
            spk_stat_path = os.path.join(os.path.dirname(gpt_ckpt_path), "spk_stat.pt")
            assert os.path.exists(
                spk_stat_path
            ), f"Missing spk_stat.pt: {spk_stat_path}"
            self.pretrain_models["spk_stat"] = torch.load(
                spk_stat_path, map_location=map_location
            ).to(device=device, dtype=dtype)
            self.logger.log(logging.INFO, "gpt loaded.")

        if decoder_config_path:
            cfg = OmegaConf.load(decoder_config_path)
            decoder = DVAE(**cfg).to(device=device, dtype=dtype_decoder).eval()
            assert decoder_ckpt_path, "decoder_ckpt_path should not be None"
            decoder.load_state_dict(
                torch.load(decoder_ckpt_path, map_location=map_location)
            )
            self.pretrain_models["decoder"] = decoder
            self.logger.log(logging.INFO, "decoder loaded.")

        if tokenizer_path:
            tokenizer = torch.load(tokenizer_path, map_location=map_location)
            tokenizer.padding_side = "left"
            self.pretrain_models["tokenizer"] = tokenizer
            self.logger.log(logging.INFO, "tokenizer loaded.")

        self.check_model()

    def _infer(
        self,
        text,
        skip_refine_text=False,
        refine_text_only=False,
        params_refine_text={},
        params_infer_code={"prompt": "[speed_5]"},
        use_decoder=True,
        stream=False,
        stream_text=False,
    ):

        # assert self.check_model(use_decoder=use_decoder)

        if not isinstance(text, list):
            text = [text]

        for i, t in enumerate(text):
            reserved_tokens = self.pretrain_models[
                "tokenizer"
            ].additional_special_tokens
            invalid_characters = count_invalid_characters(
                t, reserved_tokens=reserved_tokens
            )
            if len(invalid_characters):
                self.logger.log(
                    logging.WARNING, f"Invalid characters found! : {invalid_characters}"
                )
                text[i] = apply_character_map(t, reserved_tokens=reserved_tokens)

        if not skip_refine_text:
            text_tokens_gen = refine_text(
                self.pretrain_models, text, stream=stream, **params_refine_text
            )

            def decode_text(text_tokens):
                text_tokens = [
                    i[
                        i
                        < self.pretrain_models["tokenizer"].convert_tokens_to_ids(
                            "[break_0]"
                        )
                    ]
                    for i in text_tokens
                ]
                text = self.pretrain_models["tokenizer"].batch_decode(text_tokens)
                return text

            if stream_text:
                for result in text_tokens_gen:
                    text_incomplete = decode_text(result["ids"])
                    if refine_text_only and stream:
                        yield text_incomplete
                    if refine_text_only:
                        return
            else:
                result = next(text_tokens_gen)
                text = decode_text(result["ids"])
                if refine_text_only:
                    yield text
            if refine_text_only:
                return

        text = [params_infer_code.get("prompt", "") + i for i in text]
        params_infer_code.pop("prompt", "")
        result_gen = infer_code(
            self.pretrain_models,
            text,
            **params_infer_code,
            return_hidden=use_decoder,
            stream=stream,
        )
        if use_decoder:
            field = "hiddens"
            docoder_name = "decoder"
        else:
            field = "ids"
            docoder_name = "dvae"
        vocos_decode = lambda spec: [
            self.pretrain_models["vocos"]
            .decode(i.cpu() if torch.backends.mps.is_available() else i)
            .cpu()
            .numpy()
            for i in spec
        ]
        if stream:

            length = 0
            for result in result_gen:
                chunk_data = result[field][0]
                assert len(result[field]) == 1
                start_seek = length
                length = len(chunk_data)
                self.logger.debug(
                    f"{start_seek=} total len: {length}, new len: {length - start_seek = }"
                )
                chunk_data = chunk_data[start_seek:]
                if not len(chunk_data):
                    continue
                self.logger.debug(f"new hidden {len(chunk_data)=}")
                mel_spec = [
                    self.pretrain_models[docoder_name](i[None].permute(0, 2, 1))
                    for i in [chunk_data]
                ]
                wav = vocos_decode(mel_spec)
                self.logger.debug(f"yield wav chunk {len(wav[0])=} {len(wav[0][0])=}")
                yield wav
            return
        mel_spec = [
            self.pretrain_models[docoder_name](i[None].permute(0, 2, 1))
            for i in next(result_gen)[field]
        ]
        yield vocos_decode(mel_spec)

    def infer(
        self,
        text,
        skip_refine_text=False,
        refine_text_only=False,
        params_refine_text={},
        params_infer_code={"prompt": "[speed_5]"},
        use_decoder=True,
        stream=False,
    ):
        res_gen = self._infer(
            text=text,
            skip_refine_text=skip_refine_text,
            refine_text_only=refine_text_only,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            use_decoder=use_decoder,
            stream=stream,
        )
        if stream:
            return res_gen
        else:
            return next(res_gen)

    def refiner_prompt(self, text, params_refine_text={}, stream=False):
        return self.infer(
            text=text,
            skip_refine_text=False,
            refine_text_only=True,
            params_refine_text=params_refine_text,
            stream=stream,
        )

    def generate_audio(
        self,
        prompt,
        params_infer_code={"prompt": "[speed_5]"},
        use_decoder=True,
        stream=False,
    ):
        return self.infer(
            prompt,
            skip_refine_text=True,
            params_infer_code=params_infer_code,
            use_decoder=use_decoder,
            stream=stream,
        )

    def sample_random_speaker(
        self,
    ) -> torch.Tensor:
        assert self.pretrain_models["gpt"] is not None, "gpt model not loaded"
        dim = self.pretrain_models["gpt"].gpt.layers[0].mlp.gate_proj.in_features
        std, mean = self.pretrain_models["spk_stat"].chunk(2)
        return torch.randn(dim, device=std.device) * std + mean
