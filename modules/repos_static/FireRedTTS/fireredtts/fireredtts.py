import os
import json
import torch
from fireredtts.modules.gpt.gpt import GPT
from fireredtts.modules import Token2Wav, MelSpectrogramExtractor
from fireredtts.modules.tokenizer.tokenizer import VoiceBpeTokenizer
from fireredtts.modules.codec.speaker import SpeakerEmbedddingExtractor
from fireredtts.utils.utils import load_audio

import time


class FireRedTTS:
    def __init__(self, config_path, pretrained_path, device="cuda"):
        self.device = device
        self.config = json.load(open(config_path))
        self.gpt_path = os.path.join(pretrained_path, "fireredtts_gpt.pt")
        self.token2wav_path = os.path.join(pretrained_path, "fireredtts_token2wav.pt")
        self.speaker_extractor_path = os.path.join(
            pretrained_path, "fireredtts_speaker.bin"
        )
        assert os.path.exists(self.token2wav_path)
        assert os.path.exists(self.gpt_path)
        assert os.path.exists(self.speaker_extractor_path)

        # tokenizer;
        self.text_tokenizer = VoiceBpeTokenizer()

        # speaker ectractor
        self.speaker_extractor = SpeakerEmbedddingExtractor(
            ckpt_path=self.speaker_extractor_path, device=device
        )

        # load gpt model
        self.gpt = GPT(
            start_text_token=self.config["gpt"]["gpt_start_text_token"],
            stop_text_token=self.config["gpt"]["gpt_stop_text_token"],
            layers=self.config["gpt"]["gpt_layers"],
            model_dim=self.config["gpt"]["gpt_n_model_channels"],
            heads=self.config["gpt"]["gpt_n_heads"],
            max_text_tokens=self.config["gpt"]["gpt_max_text_tokens"],
            max_mel_tokens=self.config["gpt"]["gpt_max_audio_tokens"],
            max_prompt_tokens=self.config["gpt"]["gpt_max_prompt_tokens"],
            code_stride_len=self.config["gpt"]["gpt_code_stride_len"],
            number_text_tokens=self.config["gpt"]["gpt_number_text_tokens"],
            num_audio_tokens=self.config["gpt"]["gpt_num_audio_tokens"],
            start_audio_token=self.config["gpt"]["gpt_start_audio_token"],
            stop_audio_token=self.config["gpt"]["gpt_stop_audio_token"],
        )

        sd = torch.load(self.gpt_path, map_location=device)["model"]
        self.gpt.load_state_dict(sd, strict=True)
        self.gpt = self.gpt.to(device=device)
        self.gpt.eval()
        self.gpt.init_gpt_for_inference(kv_cache=True)

        # mel-spectrogram extractor
        self.mel_extractor = MelSpectrogramExtractor()

        # load token2wav model
        self.token2wav = Token2Wav.init_from_config(self.config)
        sd = torch.load(self.token2wav_path, map_location="cpu")
        self.token2wav.load_state_dict(sd, strict=True)
        self.token2wav.generator.remove_weight_norm()
        self.token2wav.eval()
        self.token2wav = self.token2wav.to(device)

    def extract_spk_embeddings(self, prompt_wav):
        _, _, audio_resampled = load_audio(audiopath=prompt_wav, sampling_rate=16000)
        audio_len = torch.tensor(
            data=[audio_resampled.shape[1]], dtype=torch.long, requires_grad=False
        )

        # speaker embeddings [1,512]
        spk_embeddings = self.speaker_extractor(
            audio_resampled.to(device="cuda")
        ).unsqueeze(0)

        return spk_embeddings

    def do_gpt_inference(self, spk_gpt, text_tokens):
        """_summary_

        Args:
            spk_gpt (_type_): speaker embeddidng in gpt
            text_tokens (_type_): text tokens
        """
        with torch.no_grad():
            gpt_codes = self.gpt.generate(
                cond_latents=spk_gpt,
                text_inputs=text_tokens,
                input_tokens=None,
                do_sample=True,
                top_p=0.85,
                top_k=30,
                temperature=0.75,
                num_return_sequences=9,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=2.0,
                output_attentions=False,
            )

        seqs = []
        EOS_TOKEN = self.config["gpt"]["gpt_stop_audio_token"]
        for seq in gpt_codes:
            index = (seq == EOS_TOKEN).nonzero(as_tuple=True)[0][0]
            seq = seq[:index]
            seqs.append(seq)

        sorted_seqs = sorted(seqs, key=lambda i: len(i), reverse=False)
        gpt_codes = sorted_seqs[2].unsqueeze(0)  # [1, len]
        # sorted_len = [len(l) for l in sorted_seqs]
        # print("---sorted_len:", sorted_len)

        return gpt_codes

    def synthesize(self, prompt_wav, text, lang="auto"):
        """_summary_

        Args:
            prompts_wav (_type_): prompts_wav path
            text (_type_): text
            lang (_type_): language of text
        """
        # Currently only supports Chinese and English
        assert lang in ["zh", "en", "auto"]
        assert os.path.exists(prompt_wav)

        # text to tokens
        text_tokens = self.text_tokenizer.encode(text=text, lang=lang)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0).to(self.device)
        assert text_tokens.shape[-1] < 400

        # extract speaker embedding
        spk_embeddings = self.extract_spk_embeddings(prompt_wav=prompt_wav).unsqueeze(0)
        with torch.no_grad():
            spk_gpt = self.gpt.reference_embedding(spk_embeddings)

        # gpt inference
        gpt_start_time = time.time()
        gpt_codes = self.do_gpt_inference(spk_gpt=spk_gpt, text_tokens=text_tokens)
        gpt_end_time = time.time()
        gpt_dur = gpt_end_time - gpt_start_time

        # prompt mel-spectrogram compute
        prompt_mel = (
            self.mel_extractor(wav_path=prompt_wav).unsqueeze(0).to(self.device)
        )
        # convert token to waveform (b=1, t)
        voc_start_time = time.time()
        rec_wavs = self.token2wav.inference(gpt_codes, prompt_mel, n_timesteps=10)
        voc_end_time = time.time()
        voc_dur = voc_end_time - voc_start_time
        all_dur = voc_end_time - gpt_start_time

        # rtf compute
        # audio_dur = rec_wavs.shape[-1] / 24000
        # rtf_gpt = gpt_dur / audio_dur
        # rtf_voc = voc_dur / audio_dur
        # rtf_all = all_dur / audio_dur

        return rec_wavs
