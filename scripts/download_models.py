import logging

try:
    logging.basicConfig(level=logging.INFO)
except:
    pass

from scripts.dl_args import parser_args
from scripts.dl_chattts import ChatTTSDownloader
from scripts.dl_enhance import ResembleEnhanceDownloader
from scripts.dl_cosyvoice import CosyVoiceDownloader
from scripts.dl_cosyvoice_base import CosyVoiceBaseDownloader
from scripts.dl_cosyvoice_instruct import CosyVoiceInstructDownloader
from scripts.dl_f5_tts import F5TTSDownloader
from scripts.dl_faster_whisper import FasterWhisperDownloader
from scripts.dl_fire_red_tts import FireRedTTSDownloader
from scripts.dl_fish_speech_1_2sft import FishSpeechDownloader
from scripts.dl_fish_speech_1_4 import FishSpeech14Downloader
from scripts.dl_open_voice import OpenVoiceDownloader
from scripts.dl_vocos_mel_24khz import VocosMel24khzDownloader
from scripts.dl_index_tts import IndexTTSDownloader

from scripts.ModelDownloader import ModelDownloader


def main():
    args = parser_args()

    downloaders: list[ModelDownloader] = []
    downloaders.append(ChatTTSDownloader())
    downloaders.append(ResembleEnhanceDownloader())
    downloaders.append(CosyVoiceBaseDownloader())
    downloaders.append(CosyVoiceInstructDownloader())
    downloaders.append(F5TTSDownloader())
    downloaders.append(FasterWhisperDownloader())
    downloaders.append(FireRedTTSDownloader())
    downloaders.append(FishSpeechDownloader())
    downloaders.append(FishSpeech14Downloader())
    downloaders.append(OpenVoiceDownloader())
    downloaders.append(VocosMel24khzDownloader())
    downloaders.append(IndexTTSDownloader())

    for downloader in downloaders:
        downloader(source=args.source)


if __name__ == "__main__":
    main()
