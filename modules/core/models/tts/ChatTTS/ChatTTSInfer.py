import logging
from dataclasses import dataclass, is_dataclass
from typing import Generator, List, Union

import numpy as np
import torch

from modules import config
from modules.devices import devices
from modules.repos_static.ChatTTS.ChatTTS.core import Chat
from modules.repos_static.ChatTTS.ChatTTS.model import GPT
from modules.utils.monkey_tqdm import disable_tqdm


def del_all(d: Union[dict, list]):
    if is_dataclass(d):
        for k in list(vars(d).keys()):
            x = getattr(d, k)
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
            delattr(d, k)
    elif isinstance(d, dict):
        lst = list(d.keys())
        for k in lst:
            x = d.pop(k)
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
    elif isinstance(d, list):
        while len(d):
            x = d.pop()
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
    else:
        del d


class ChatTTSInfer:
    model_id = "chat-tts"

    logger = logging.getLogger(__name__)

    current_infer = None

    def __init__(self, instance: Chat) -> None:
        self.instance = instance
        self.device = instance.device
        self.dtype = devices.dtype
        ChatTTSInfer.current_infer = self

        if config.runtime_env_vars.debug_generate:
            self.logger.setLevel(logging.DEBUG)

    def get_tokenizer(self):
        return self.instance.tokenizer._tokenizer

    @classmethod
    def interrupt(cls):
        # FIXME: 目前没法立即停止，会等到下一个chunk？好像得改 `gpt.py`
        if cls.current_infer:
            cls.current_infer.instance.interrupt()
            cls.logger.info("Interrupted current infer")

    @torch.inference_mode()
    def _sample_audio_speaker(
        self, wav: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        wav = wav.to(device=self.device, dtype=self.dtype)
        # TODO: 最好不要 autocast ，但是得改 dvae 的代码
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            return (
                self.instance.dvae(wav, "encode")
                .squeeze_(0)
                .to(device=self.device, dtype=self.dtype)
            )

    def infer(
        self,
        text: Union[str, list[str]],
        stream=False,
        skip_refine_text=False,
        refine_text_only=False,
        use_decoder=True,
        params_refine_text=Chat.RefineTextParams(),
        params_infer_code=Chat.InferCodeParams(),
    ):
        self.instance.context.set(False)
        res_gen = self._infer(
            text=text,
            stream=stream,
            skip_refine_text=skip_refine_text,
            refine_text_only=refine_text_only,
            use_decoder=use_decoder,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
        )
        if stream:
            return res_gen
        else:
            return next(res_gen)

    @torch.inference_mode()
    def _infer(
        self,
        text: Union[str, list[str]],
        stream=False,
        skip_refine_text=False,
        refine_text_only=False,
        use_decoder=True,
        params_refine_text=Chat.RefineTextParams(),
        params_infer_code=Chat.InferCodeParams(),
    ):
        if not isinstance(text, list):
            text = [text]

        # NOTE: 作用就是尽量不让 vocos 处理短序列 (但是可能导致略微性能降低)
        # 但是效果不太好...暂时关闭
        # smooth_decoding = stream
        smooth_decoding = False

        self.logger.debug(
            f"Start infer: stream={stream}, skip_refine_text={skip_refine_text}, refine_text_only={refine_text_only}, use_decoder={use_decoder}, smooth_decoding={smooth_decoding}"
        )
        self.logger.debug(
            f"params_refine_text={params_refine_text}, params_infer_code={params_infer_code}"
        )
        self.logger.debug(f"Text: {text}")

        with torch.no_grad():

            if not skip_refine_text:
                refined = self.instance._refine_text(
                    text,
                    self.instance.device,
                    params_refine_text,
                )
                text_tokens = refined.ids
                text_tokens = [
                    i[i.less(self.instance.tokenizer.break_0_ids)] for i in text_tokens
                ]
                text = self.instance.tokenizer.decode(text_tokens)
                refined.destroy()
                if refine_text_only:
                    yield text
                    return
            if not smooth_decoding:
                length = [0 for _ in range(len(text))]
                for result in self.instance._infer_code(
                    text,
                    stream,
                    self.instance.device,
                    use_decoder,
                    params_infer_code,
                ):
                    wavs = self._decode_to_wavs(result, length, use_decoder)
                    result.destroy()
                    yield wavs
            else:
                # NOTE: 貌似没什么用...?
                # smooth_decoding 即使用了滑动窗口的解码，每次都保留上一段的隐藏状态一起解码，并且保留上一段的音频长度用于截取
                @dataclass(repr=False, eq=False)
                class WavWindow:
                    start_seek: int = 0
                    prev_token_len: int = 0
                    prev_wav_len: int = 0

                wavs_windows = [WavWindow() for _ in range(len(text))]

                overlap = 1024
                for result in self.instance._infer_code(
                    text,
                    stream,
                    self.instance.device,
                    use_decoder,
                    params_infer_code,
                ):
                    # method1: 滑动
                    wavs = []
                    for i, window in enumerate(wavs_windows):
                        x = result.hiddens[i] if use_decoder else result.ids[i]
                        start_seek = window.start_seek
                        prev_token_len = window.prev_token_len
                        prev_wav_len = window.prev_wav_len

                        length = len(x)
                        if length <= start_seek:
                            wavs.append(None)
                            continue

                        chunk_data = x[start_seek - prev_token_len :]

                        decoder = (
                            self.instance.decoder if use_decoder else self.instance.dvae
                        )
                        input_data = (
                            chunk_data[None]
                            .permute(0, 2, 1)
                            .to(device=self.instance.device)
                        )
                        if use_decoder:
                            input_data = input_data.to(dtype=self.instance.dtype)
                        mel_spec = decoder(input_data)

                        # print(chunk_data.shape, mel_spec.shape)
                        del input_data
                        del chunk_data

                        wav = self.instance._vocos_decode(mel_spec)
                        start_wav = prev_wav_len - overlap
                        start_wav = max(0, start_wav)
                        wav = wav[:, start_wav:-overlap]
                        wav_len = len(wav[0])
                        del_all(mel_spec)

                        wavs.append(wav)

                        # Update window information
                        window.prev_token_len = length - start_seek
                        window.start_seek = length
                        window.prev_wav_len = wav_len

                        # print(wav.shape)
                        # print(
                        #     {
                        #         i: window.start_seek
                        #         for i, window in enumerate(wavs_windows)
                        #     }
                        # )
                        # print(
                        #     {
                        #         i: window.prev_token_len
                        #         for i, window in enumerate(wavs_windows)
                        #     }
                        # )

                    result.destroy()
                    del_all(x)
                    yield wavs

                    # mothed2: 全部编码，并截切出当前的部分，而不是滚动
                    # length = [0 for _ in range(len(text))]
                    # wavs = self._decode_to_wavs(result, length, use_decoder)
                    # cut_wavs = [[] for _ in range(len(text))]
                    # # 更新window
                    # for i, win in enumerate(wavs_windows):
                    #     cut_wavs[i] = wavs[i][:, win.prev_wav_len :]
                    #     win.prev_wav_len = len(wavs[i][0])
                    # yield cut_wavs

                    pass

    def _decode_to_wavs(
        self, result: GPT.GenerationOutputs, start_seeks: List[int], use_decoder: bool
    ):
        x = result.hiddens if use_decoder else result.ids
        wavs: List[np.ndarray] = []
        for i, chunk_data in enumerate(x):
            start_seek = start_seeks[i]
            length = len(chunk_data)
            if length <= start_seek:
                wavs.append(None)
                continue
            start_seeks[i] = length
            chunk_data = chunk_data[start_seek:]
            decoder = self.instance.decoder if use_decoder else self.instance.dvae
            input_data = (
                chunk_data[None].permute(0, 2, 1).to(device=self.instance.device)
            )
            if use_decoder:
                input_data = input_data.to(dtype=self.dtype)
            mel_spec = decoder(input_data)
            del input_data
            del chunk_data
            wavs.append(self.instance._vocos_decode(mel_spec))
            del_all(mel_spec)
        result.destroy()
        del_all(x)
        return wavs

    def _generate_audio(
        self,
        text: Union[str, list[str]],
        spk_emb: Union[None, torch.Tensor] = None,
        spk_smp: Union[None, torch.Tensor] = None,
        txt_smp: Union[None, str] = None,
        top_P=0.7,
        top_K=20,
        temperature=0.3,
        repetition_penalty=1.05,
        max_new_token=2048,
        prompt="",
        prompt1="",
        prompt2="",
        prefix="",
        stream_chunk_size=96,
        use_decoder=True,
        stream=False,
    ):
        params = Chat.InferCodeParams(
            spk_emb=spk_emb,
            spk_smp=spk_smp,
            txt_smp=txt_smp,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
            prompt=prompt,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
            stream_batch=stream_chunk_size,
            ensure_non_empty=False,
        )
        return self.infer(
            text=text,
            stream=stream,
            skip_refine_text=True,
            params_infer_code=params,
            use_decoder=use_decoder,
        )

    def generate_audio(
        self,
        text: Union[str, list[str]],
        spk_emb: Union[None, torch.Tensor] = None,
        spk_smp: Union[None, torch.Tensor] = None,
        txt_smp: Union[None, str] = None,
        top_P=0.7,
        top_K=20,
        temperature=0.3,
        repetition_penalty=1.05,
        max_new_token=2048,
        prompt="",
        prompt1="",
        prompt2="",
        prefix="",
        use_decoder=True,
    ) -> list[np.ndarray]:
        with disable_tqdm(enabled=config.runtime_env_vars.off_tqdm):
            data = self._generate_audio(
                text=text,
                spk_emb=spk_emb,
                spk_smp=spk_smp,
                txt_smp=txt_smp,
                top_P=top_P,
                top_K=top_K,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_token=max_new_token,
                prompt=prompt,
                prompt1=prompt1,
                prompt2=prompt2,
                prefix=prefix,
                use_decoder=use_decoder,
                stream=False,
            )
            data = [i for i in data if i is not None]
            return data

    def generate_audio_stream(
        self,
        text: Union[str, list[str]],
        spk_emb: Union[None, torch.Tensor] = None,
        spk_smp: Union[None, torch.Tensor] = None,
        txt_smp: Union[None, str] = None,
        top_P=0.7,
        top_K=20,
        temperature=0.3,
        repetition_penalty=1.05,
        max_new_token=2048,
        prompt="",
        prompt1="",
        prompt2="",
        prefix="",
        stream_chunk_size=96,
        use_decoder=True,
    ) -> Generator[list[np.ndarray], None, None]:
        gen: Generator[list[np.ndarray], None, None] = self._generate_audio(
            text=text,
            spk_emb=spk_emb,
            spk_smp=spk_smp,
            txt_smp=txt_smp,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
            prompt=prompt,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
            use_decoder=use_decoder,
            stream_chunk_size=stream_chunk_size,
            stream=True,
        )

        def _generator():
            with disable_tqdm(enabled=config.runtime_env_vars.off_tqdm):
                for audio_arr in gen:
                    # 如果为空就用空 array 填充
                    # NOTE: 因为长度不一定，所以某个位置可能是 None
                    audio_arr = [np.empty(0) if i is None else i for i in audio_arr]
                    yield audio_arr

        return _generator()

    def _refine_text(
        self,
        text: Union[str, list[str]],
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
            return self.infer(
                text=text,
                stream=stream,
                skip_refine_text=False,
                refine_text_only=True,
                params_refine_text=params,
                use_decoder=False,
            )

    def refine_text(
        self,
        text: Union[str, list[str]],
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
        text: Union[str, list[str]],
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
