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
  - **Whisper**: Use the Whisper model for high-quality speech-to-text (ASR)
  - **SenseVoice**: ASR model in development, coming soon

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

| Model Category   | Model Name                                                                                  | Streaming Level | Multi-Language Support       | Status                  |
| ---------------- | ------------------------------------------------------------------------------------------- | --------------- | ---------------------------- | ----------------------- |
| **TTS**          | [ChatTTS](https://github.com/2noise/ChatTTS)                                                | token-level     | en, zh                       | ✅                       |
|                  | [FishSpeech](https://github.com/fishaudio/fish-speech)                                       | sentence-level  | en, zh, jp, ko           | ✅ (1.4) |
|                  | [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)                                        | sentence-level  | en, zh, jp, yue, ko          | ✅                       |
|                  | [FireRedTTS](https://github.com/FireRedTeam/FireRedTTS)                                      | sentence-level  | en, zh                       | ✅                       |
|                  | [F5-TTS](https://github.com/SWivid/F5-TTS)                                                  | sentence-level  | en, zh                       | ✅                       |
|                  | GPTSoVits                                                                                    | sentence-level  |                              | 🚧                       |
| **ASR**          | [Whisper](https://github.com/openai/whisper)                                                | 🚧              | ✅                           | ✅                       |
|                  | [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)                                      | 🚧              | ✅                           | 🚧                       |
| **Voice Clone**  | [OpenVoice](https://github.com/myshell-ai/OpenVoice)                                        |                 |                              | ✅                       |
|                  | [RVC](https://github.com/svc-develop-team/RVC)                                              |                 |                              | 🚧                       |
| **Enhancer**     | [ResembleEnhance](https://github.com/resemble-ai/resemble-enhance)                          |                 |                              | ✅                       |

## Model Download

Since Forge primarily focuses on API functionality development, automatic download logic has not yet been implemented. To download models, you need to manually invoke the download scripts, which can be found in the `./scripts` directory.

### Download Script

| Function     | Model          | Download Command                                                          |
| ------------ | -------------- | ------------------------------------------------------------------------- |
| **TTS**      | ChatTTS        | `python -m scripts.dl_chattts --source huggingface`                       |
|              | FishSpeech(1.4)     | `python -m scripts.downloader.fish_speech_1_4 --source huggingface`    |
|              | CosyVoice      | `python -m scripts.downloader.dl_cosyvoice_instruct --source huggingface` |
|              | FireRedTTS     | `python -m scripts.downloader.fire_red_tts --source huggingface`          |
| **ASR**      | Whisper        | `python -m scripts.downloader.faster_whisper --source huggingface`        |
| **CV**       | OpenVoice      | `python -m scripts.downloader.open_voice --source huggingface`            |
| **Enhancer** | Enhancer Model | `python -m scripts.dl_enhance --source huggingface`                       |

> **Note**: If you need to use ModelScope to download models, use `--source modelscope`. Some models may not be available for download using ModelScope.

> **About CosyVoice**: It's unclear which model to use. Overall, the `instruct` model has the most features, but its quality may not be the best. If you wish to use other models, feel free to select `dl_cosyvoice_base.py`, `dl_cosyvoice_instruct.py`, or the `sft` script. The loading priority is `base` > `instruct` > `sft`, and you can determine which to load based on folder existence.

## FAQ

### How to perform voice cloning?

Currently, voice cloning is supported across various models, and formats like reference audio in `skpv1` are also adapted. Here are a few methods to use voice cloning:

1. **In the WebUI**: You can upload reference audio in the voice selection section, which is the simplest way to use the voice cloning feature.
2. **Using the API**: When using the API, you need to use a voice (i.e., a speaker) for voice cloning. First, you need to create a speaker file (e.g., `.spkv1.json`) with the required voice, and when calling the API, set the `spk` parameter to the speaker's name to enable cloning.
3. **Voice Clone**: The system now also supports voice cloning using the voice clone model. When using the API, configure the appropriate `reference` to utilize this feature. (Currently, only OpenVoice is supported for voice cloning, so there’s no need to specify the model name.)

For related discussions, see issue #118.

### The generated result with a reference audio `spk` file is full of noise?

This is likely caused by an issue with the uploaded audio configuration. You can try the following solutions:

1. **Update**: Update the code and dependency versions. Most importantly, update Gradio (it's recommended to use the latest version if possible).
2. **Process the audio**: Use ffmpeg or other software to edit the audio, convert it to mono, and then upload it. You can also try converting it to WAV format.
3. **Check the text**: Make sure there are no unsupported characters in the reference text. It's also recommended to end the reference text with a `"。"` (this is a quirk of the model 😂).
4. **Create with Colab**: Consider using the Colab environment to create the `spk` file to minimize environment-related issues.
5. **TTS Test**: Currently, in the WebUI TTS page, you can upload reference audio directly. You can first test the audio and text, make adjustments, and then generate the `spk` file.

### Can I train models?

Not at the moment. This repository mainly provides a framework for inference services. There are plans to add some training-related features, but they are not a priority.

### How can I optimize inference speed?

This repository focuses on integrating and developing engineering solutions, so model inference optimizations largely depend on upstream repositories or community implementations. If you have good optimization ideas, feel free to submit an issue or PR.

For now, the most practical optimization is to enable multiple workers. When running the `launch.py` script, you can start with the `--workers N` option to increase service throughput.

There are also other potential speed-up optimizations that are not yet fully implemented. If interested, feel free to explore:

1. **Compile**: Models support compile acceleration, which can provide around a 30% speed increase, but the compilation process is slow.
2. **Flash Attention**: Flash attention acceleration is supported (using the `--flash_attn` option), but it is still not perfect.
3. **vllm**: Not yet implemented, pending updates from upstream repositories.

### What are Prompt1 and Prompt2?

> Only for ChatTTS

Both Prompt1 and Prompt2 are system prompts, but the difference lies in their insertion points. Through testing, it was found that the current model is very sensitive to the first `[Stts]` token, so two prompts are required:

- Prompt1 is inserted before the first `[Stts]`.
- Prompt2 is inserted after the first `[Stts]`.

### What is Prefix?

> Only for ChatTTS

Prefix is mainly used to control the model's generation capabilities, similar to refine prompts in official examples. The prefix should only include special non-lexical tokens, such as `[laugh_0]`, `[oral_0]`, `[speed_0]`, `[break_0]`, etc.

### What is the difference with `_p` in the Style?

In the Style settings, those with `_p` use both prompt + prefix, while those without `_p` use only the prefix.

### Why is it so slow when `--compile` is enabled?

Since inference padding has not yet been implemented, changing the shape during each inference may trigger torch to recompile.

> For now, it’s not recommended to enable this option.

### Why is it so slow in Colab, only 2 it/s?

Please ensure that you are using a GPU instead of a CPU.

- Click on the menu bar **Edit**.
- Select **Notebook Settings**.
- Choose **Hardware Accelerator** => T4 GPU.

# Documents

find more documents from [here](./docs/readme.md)

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
- Whisper: https://github.com/openai/whisper

- ChatTTS 默认说话人: https://github.com/2noise/ChatTTS/issues/238
