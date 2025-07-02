[cn](./README.md) | [en](./README.en.md) | [Discord Server](https://discord.gg/9XnXUhAy3t)

# 🍦 Speech-AI-Forge

Speech-AI-Forge is a project developed around TTS generation model, implementing an API Server and a Gradio-based WebUI.

![banner](./docs/banner.png)

You can experience and deploy Speech-AI-Forge through the following methods:

| -                        | Description                             | Link                                                                                                                                                                  |
| ------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Online Demo**          | Deployed on HuggingFace                 | [HuggingFace Spaces](https://huggingface.co/spaces/lenML/Speech-AI-Forge)                                                                                             |
| **One-Click Start**      | Click the button to start Colab         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenML/Speech-AI-Forge/blob/main/colab.en.ipynb) |
| **Container Deployment** | See the docker section                  | [Docker](#docker)                                                                                                                                                     |
| **Local Deployment**     | See the environment preparation section | [Local Deployment](#InstallationandRunning)                                                                                                                           |

## Breaking change logs

- 250702: Support Index-TTS-1.5 [#250](https://github.com/lenML/Speech-AI-Forge/issues/250)
- 250522: Support GptSoVits [#198](https://github.com/lenML/Speech-AI-Forge/issues/198)
- 250518: Support SenseVoice ASR [#122](https://github.com/lenML/Speech-AI-Forge/issues/122)
- 250508: Support Spark-TTS [#223](https://github.com/lenML/Speech-AI-Forge/issues/223)
- 250507: Support F5TTS-TTS-v1 model [#231](https://github.com/lenML/Speech-AI-Forge/issues/231)
- 250505: Support Index-TTS [#229](https://github.com/lenML/Speech-AI-Forge/issues/229)
- 241111: Add `v2/tts` API [#187](https://github.com/lenML/Speech-AI-Forge/issues/187)
- 241109: Support fishspeech [#191](https://github.com/lenML/Speech-AI-Forge/issues/191)
- 241015: Support F5TTS v0.6.2 [#176](https://github.com/lenML/Speech-AI-Forge/issues/176)
- 241009: Support FireRedTTS [#165](https://github.com/lenML/Speech-AI-Forge/issues/165)
- 240813: Support OpenVoice [#100](https://github.com/lenML/Speech-AI-Forge/issues/100)
- 240801: Add ASR API [#92](https://github.com/lenML/Speech-AI-Forge/issues/92)
- 240723: Support CosyVoice [#90](https://github.com/lenML/Speech-AI-Forge/issues/90)

## Installation and Running

First, ensure that the [relevant dependencies](./docs/dependencies.md) have been correctly installed.

Start the application:

```
python webui.py
```

### Web UI Features

[Click here for detailed documentation with images](./docs/webui_features.md)

- **TTS (Text-to-Speech)**: Powerful TTS capabilities

  - **Speaker Switch**: Switch between different voices
    - **Built-in Voices**: Multiple built-in voices available, including `27 ChatTTS` / `7 CosyVoice` voices + `1 Reference Voice`
    - **Custom Voice Upload**: Support for uploading custom voice files and performing real-time inference
    - **Reference Voice**: Upload reference audio/text and perform TTS inference based on the reference audio
  - **Style Control**: Built-in style control options to adjust the voice tone
  - **Long Text Processing**: Support for long text inference with automatic text segmentation
    - **Batch Size**: Configure `Batch size` to speed up inference for models that support batch processing
  - **Refiner**: Native text `refiner` for `ChatTTS`, supports inference of unlimited-length text
  - **Splitter Settings**: Fine-tune splitter configuration, control splitter `eos` (end of sentence) and splitting thresholds
  - **Adjuster**: Control speech parameters like `speed`, `pitch`, and `volume`, with additional `loudness normalization` for improved output quality
  - **Voice Enhancer**: Use the `Enhancer` model to improve TTS output quality, delivering better sound
  - **Generation History**: Store the last three generated results for easy comparison
  - **Multi-model Support**: Support for multiple TTS models, including `ChatTTS`, `CosyVoice`, `FishSpeech`, `GPT-SoVITS`, and `F5-TTS`

- **SSML (Speech Synthesis Markup Language)**: Advanced TTS synthesis control

  - **Splitter**: Fine control over text segmentation for long-form content
  - **PodCast**: A tool for creating `long-form` and `multi-character` audio, ideal for blogs or scripted voice synthesis
  - **From Subtitle**: Create SSML scripts directly from subtitle files for easy TTS generation
  - **Script Editor**: New SSML script editor that allows users to export and edit SSML scripts from the Splitter (PodCast, From Subtitle) for further refinement

- **Voice Management**:

  - **Builder**: Create custom voices from ChatTTS seeds or by using reference audio
  - **Test Voice**: Upload and test custom voice files quickly
  - **ChatTTS Debugging Tools**: Specific tools for debugging `ChatTTS` voices
    - **Random Seed**: Generate random voices using a random seed to create unique sound profiles
    - **Voice Blending**: Blend voices generated from different seeds to create a new voice
  - **Voice Hub**: Select and download voices from our voice library to your local machine. Access the voice repository at [Speech-AI-Forge-spks](https://github.com/lenML/Speech-AI-Forge-spks)

- **ASR (Automatic Speech Recognition)**:

  - **ASR**: Use the Whisper/SenseVoice model for high-quality speech-to-text (ASR)
  - **Force Aligment**: The Whisper model can be used for document matching to improve recognition accuracy

- **Tools**:
  - **Post Process**: Post-processing tools for audio clipping, adjustment, and enhancement to optimize speech output

### `launch.py`: API Server

In some cases, you might not need the WebUI or require higher API throughput, in which case you can start a simple API service with this script.

To start:

```bash
python launch.py
```

Once launched, you can access `http://localhost:7870/docs` to see which API endpoints are available.

More help:

- Use `python launch.py -h` to view script parameters
- Check out the [API Documentation](./docs/api.md)

## Docker

### Image

WIP (Under development)

### Manual Build

Download models: `python -m scripts.download_models --source modelscope`

> This script will download the `chat-tts` and `enhancer` models. If you need to download other models, please refer to the `Model Download` section below.

- For the webui: `docker-compose -f ./docker-compose.webui.yml up -d`
- For the API: `docker-compose -f ./docker-compose.api.yml up -d`

Environment variable configuration:

- webui: [.env.webui](./.env.webui)
- API: [.env.api](./.env.api)

## Model Support

| Model Category  | Model Name                                                         | Streaming Level | Multi-Language Support | Status      |
| --------------- | ------------------------------------------------------------------ | --------------- | ---------------------- | ----------- |
| **TTS**         | [ChatTTS](https://github.com/2noise/ChatTTS)                       | token-level     | en, zh                 | ✅          |
|                 | [FishSpeech](https://github.com/fishaudio/fish-speech)             | sentence-level  | en, zh, jp, ko         | ✅ (1.4)    |
|                 | [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)              | sentence-level  | en, zh, jp, yue, ko    | ✅(v2)      |
|                 | [FireRedTTS](https://github.com/FireRedTeam/FireRedTTS)            | sentence-level  | en, zh                 | ✅          |
|                 | [F5-TTS](https://github.com/SWivid/F5-TTS)                         | sentence-level  | en, zh                 | ✅(v0.6/v1) |
|                 | [Index-TTS](https://github.com/index-tts/index-tts)                | sentence-level  | en, zh                 | ✅(v1/v1.5) |
|                 | [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)               | sentence-level  | en, zh                 | ✅          |
|                 | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/tree/main)     | 句子级          | en, zh, ja, ko, yue    | ✅          |
| **ASR**         | [Whisper](https://github.com/openai/whisper)                       | 🚧              | ✅                     | ✅          |
|                 | [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)            | 🚧              | ✅                     | 🚧          |
| **Voice Clone** | [OpenVoice](https://github.com/myshell-ai/OpenVoice)               |                 |                        | ✅          |
| **Enhancer**    | [ResembleEnhance](https://github.com/resemble-ai/resemble-enhance) |                 |                        | ✅          |

## Model Download

Since Forge primarily focuses on API functionality development, automatic download logic has not yet been implemented. To download models, you need to manually invoke the download scripts, which can be found in the `./scripts` directory.

### Download Script

| Function     | Model           | Download Command                                                    |
| ------------ | --------------- | ------------------------------------------------------------------- |
| **TTS**      | ChatTTS         | `python -m scripts.dl_chattts --source huggingface`                 |
|              | GPT-SoVITS(v4)  | `python -m scripts.downloader.gpt_sovits_v4 --source huggingface`   |
|              | FishSpeech(1.4) | `python -m scripts.downloader.fish_speech_1_4 --source huggingface` |
|              | CosyVoice(v2)   | `python -m scripts.downloader.cosyvoice2 --source huggingface`      |
|              | FireRedTTS      | `python -m scripts.downloader.fire_red_tts --source huggingface`    |
|              | Index-TTS-1.5   | `python -m scripts.downloader.index_tts_1_5 --source huggingface`   |
|              | Index-TTS       | `python -m scripts.downloader.index_tts --source huggingface`       |
|              | Spark-TTS       | `python -m scripts.downloader.spark_tts --source huggingface`       |
|              | F5-TTS(v0.6)    | `python -m scripts.downloader.f5_tts --source huggingface`          |
|              | F5-TTS(v1)      | `python -m scripts.downloader.f5_tts_v1 --source huggingface`       |
|              | F5-TTS(vocos)   | `python -m scripts.downloader.vocos_mel_24khz --source huggingface` |
| **ASR**      | Whisper         | `python -m scripts.downloader.faster_whisper --source huggingface`  |
| **CV**       | OpenVoice       | `python -m scripts.downloader.open_voice --source huggingface`      |
| **Enhancer** | Enhancer        | `python -m scripts.dl_enhance --source huggingface`                 |

> **Note**: If you need to use ModelScope to download models, use `--source modelscope`. Some models may not be available for download using ModelScope.

## FAQ

[Goto Discussion Page](https://github.com/lenML/Speech-AI-Forge/discussions/242)

# Documents

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/lenML/Speech-AI-Forge)

[Learn About Documents](https://github.com/lenML/Speech-AI-Forge/issues/240)

# Contributing

To contribute, clone the repository, make your changes, commit and push to your clone, and submit a pull request.

# References

- ChatTTS: https://github.com/2noise/ChatTTS
- PaddleSpeech: https://github.com/PaddlePaddle/PaddleSpeech
- resemble-enhance: https://github.com/resemble-ai/resemble-enhance
- OpenVoice: https://github.com/myshell-ai/OpenVoice
- FishSpeech: https://github.com/fishaudio/fish-speech
- SenseVoice: https://github.com/FunAudioLLM/SenseVoice
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- FireRedTTS: https://github.com/FireRedTeam/FireRedTTS
- F5-TTS: https://github.com/SWivid/F5-TTS
- Index-TTS: https://github.com/index-tts/index-tts
- Spark-TTS: https://github.com/SparkAudio/Spark-TTS
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS

- Whisper: https://github.com/openai/whisper

- ChatTTS 默认说话人: https://github.com/2noise/ChatTTS/issues/238
