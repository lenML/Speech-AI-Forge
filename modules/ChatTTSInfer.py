from typing import Generator, Union

import numpy as np
import torch
from modules.ChatTTS.ChatTTS.core import Chat
from modules.utils.monkey_tqdm import disable_tqdm
from modules import config


# 主要解决类型问题
class ChatTTSInfer:

    def __init__(self, instance: Chat) -> None:
        self.instance = instance

    def _generate_audio(
        self,
        text: str,
        spk_emb: Union[None, torch.Tensor] = None,
        top_P=0.7,
        top_K=20,
        temperature=0.3,
        repetition_penalty=1.05,
        max_new_token=2048,
        prompt1="",
        prompt2="",
        prefix="",
        use_decoder=True,
        stream=False,
    ):
        params = Chat.InferCodeParams(
            prompt="",
            spk_emb=spk_emb,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
        )
        return self.instance.infer(
            text=text,
            stream=stream,
            skip_refine_text=True,
            params_infer_code=params,
            do_text_normalization=False,
            do_homophone_replacement=False,
            use_decoder=use_decoder,
        )

    def generate_audio(
        self,
        text: str,
        spk_emb: Union[None, torch.Tensor] = None,
        top_P=0.7,
        top_K=20,
        temperature=0.3,
        repetition_penalty=1.05,
        max_new_token=2048,
        prompt1="",
        prompt2="",
        prefix="",
        use_decoder=True,
    ) -> list[np.ndarray]:
        with disable_tqdm(enabled=config.runtime_env_vars.off_tqdm):
            return self._generate_audio(
                text=text,
                spk_emb=spk_emb,
                top_P=top_P,
                top_K=top_K,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_token=max_new_token,
                prompt1=prompt1,
                prompt2=prompt2,
                prefix=prefix,
                use_decoder=use_decoder,
                stream=False,
            )

    def generate_audio_stream(
        self,
        text: str,
        spk_emb=None,
        top_P=0.7,
        top_K=20,
        temperature=0.3,
        repetition_penalty=1.05,
        max_new_token=2048,
        prompt1="",
        prompt2="",
        prefix="",
        use_decoder=True,
    ) -> Generator[list[np.ndarray], None, None]:
        gen = self._generate_audio(
            text=text,
            spk_emb=spk_emb,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
            use_decoder=use_decoder,
            stream=True,
        )

        def _generator():
            with disable_tqdm(enabled=config.runtime_env_vars.off_tqdm):
                for audio in gen:
                    yield audio

        return _generator()

    def _refine_text(
        self,
        text: str,
        top_P=0.7,
        top_K=20,
        temperature=0.7,
        repetition_penalty=1.0,
        max_new_token=384,
        prompt="",
        stream=False,
    ):
        params = Chat.RefineTextParams(
            prompt=prompt,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
        )
        with disable_tqdm(enabled=config.runtime_env_vars.off_tqdm):
            return self.instance.infer(
                text=text,
                stream=stream,
                skip_refine_text=False,
                refine_text_only=True,
                params_refine_text=params,
                do_text_normalization=False,
                do_homophone_replacement=False,
                use_decoder=False,
            )

    def refine_text(
        self,
        text: str,
        top_P=0.7,
        top_K=20,
        temperature=0.7,
        repetition_penalty=1.0,
        max_new_token=384,
        prompt="",
    ) -> str:
        return self._refine_text(
            text=text,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
            prompt=prompt,
            stream=False,
        )

    def refine_text_stream(
        self,
        text: str,
        top_P=0.7,
        top_K=20,
        temperature=0.7,
        repetition_penalty=1.0,
        max_new_token=384,
        prompt="",
    ) -> Generator[str, None, None]:
        return self._refine_text(
            text=text,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
            prompt=prompt,
            stream=True,
        )
