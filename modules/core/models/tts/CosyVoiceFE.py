import os
from typing import Callable

import inflect
import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper

from modules.devices import devices


class CosyVoiceFrontEnd:

    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = "",
        instruct: bool = False,
        allowed_special: str = "all",
        device: torch.device = torch.cuda,
        dtype: torch.dtype = torch.float32,
    ):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = device
        self.dtype = dtype
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=[
                (
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available()
                    else "CPUExecutionProvider"
                )
            ],
        )
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location="cpu")
        self.instruct = instruct
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(
            device=self.device
        )
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return text_token, text_token_len

    def _extract_speech_token(self, speech):
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = (
            self.speech_tokenizer_session.run(
                None,
                {
                    self.speech_tokenizer_session.get_inputs()[0]
                    .name: feat.detach()
                    .cpu()
                    .numpy(),
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array(
                        [feat.shape[2]], dtype=np.int32
                    ),
                },
            )[0]
            .flatten()
            .tolist()
        )
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(
            device=self.device
        )
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        embedding = torch.tensor([embedding]).to(device=self.device, dtype=self.dtype)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = (
            self.feat_extractor(speech)
            .squeeze(dim=0)
            .transpose(0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_feat, speech_feat_len

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_speech_16k, resample_rate
    ):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=resample_rate
        )(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = (
                speech_feat[:, : 2 * token_len],
                2 * token_len,
            )
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "prompt_text": prompt_text_token,
            "prompt_text_len": prompt_text_token_len,
            "llm_prompt_speech_token": speech_token,
            "llm_prompt_speech_token_len": speech_token_len,
            "flow_prompt_speech_token": speech_token,
            "flow_prompt_speech_token_len": speech_token_len,
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": speech_feat_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(
            tts_text, "", prompt_speech_16k, resample_rate
        )
        # in cross lingual mode, we remove prompt in llm
        del model_input["prompt_text"]
        del model_input["prompt_text_len"]
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input["llm_embedding"]
        instruct_text_token, instruct_text_token_len = self._extract_text_token(
            instruct_text + "<endofprompt>"
        )
        model_input["prompt_text"] = instruct_text_token
        model_input["prompt_text_len"] = instruct_text_token_len
        return model_input

    def frontend_instruct2(
        self, tts_text, instruct_text, prompt_speech_16k, resample_rate
    ):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(
            instruct_text + "<|endofprompt|>"
        )
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=resample_rate
        )(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = (
                speech_feat[:, : 2 * token_len],
                2 * token_len,
            )
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "prompt_text": prompt_text_token,
            "prompt_text_len": prompt_text_token_len,
            "flow_prompt_speech_token": speech_token,
            "flow_prompt_speech_token_len": speech_token_len,
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": speech_feat_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(
            prompt_speech_16k
        )
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=resample_rate
        )(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(
            prompt_speech_resample
        )
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(
            source_speech_16k
        )
        model_input = {
            "source_speech_token": source_speech_token,
            "source_speech_token_len": source_speech_token_len,
            "flow_prompt_speech_token": prompt_speech_token,
            "flow_prompt_speech_token_len": prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "flow_embedding": embedding,
        }
        return model_input
