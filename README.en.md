[cn](./README.md) | [en](./README.en.md) | [Discord Server](https://discord.gg/9XnXUhAy3t)

# 🍦 ChatTTS-Forge

ChatTTS-Forge is a project developed around TTS generation model, implementing an API Server and a Gradio-based WebUI.

![banner](./docs/banner.png)

You can experience and deploy ChatTTS-Forge through the following methods:

| -                        | Description                             | Link                                                                                                                                                                |
| ------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Online Demo**          | Deployed on HuggingFace                 | [HuggingFace Spaces](https://huggingface.co/spaces/lenML/ChatTTS-Forge)                                                                                             |
| **One-Click Start**      | Click the button to start Colab         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenML/ChatTTS-Forge/blob/main/colab.en.ipynb) |
| **Container Deployment** | See the docker section                  | [Docker](#docker)                                                                                                                                                   |
| **Local Deployment**     | See the environment preparation section | [Local Deployment](#InstallationandRunning)                                                                                                                         |

## Installation and Running

First, ensure that the [relevant dependencies](./docs/dependencies.md) have been correctly installed.

Start the application:

```
python webui.py
```

### WebUI Features

[Click here for a detailed graphical introduction](./docs/webui_features.md)

- TTS: Functions related to the TTS model
  - Speaker Switch: Allows you to switch between different voices
    - Built-in voices: Several built-in voices are available, including `27 ChatTTS` / `7 CosyVoice` voices + `1 reference voice`
    - Voice upload: Custom voice files can be uploaded, enabling real-time inference
    - Reference voice: Upload reference audio/text and directly use the reference audio for `tts` inference
  - Style: Includes various built-in style controls
  - Long Text: Supports long text inference with automatic text splitting
    - Batch Size: You can set the `Batch size`, which speeds up long text inference for models that support `batch` inference
  - Refiner: Supports the `ChatTTS` native text `refiner` and can handle infinitely long texts
  - Splitter: Allows configuration of the splitter to control `eos` and `split thresholds`
  - Adjuster: Adjusts `speed/pitch/volume` with additional useful features like `loudness equalization`
  - Voice Enhancer: Enhances `TTS` output with the `Enhancer` model to further improve output quality
  - Generation History: Keeps the last three generated results for easy comparison
  - Multi-model: Supports multiple `TTS` models for inference, including `ChatTTS` / `CosyVoice` / `FishSpeech` / `GPT-SoVITS`
- SSML: An advanced TTS synthesis control tool with XML-like syntax
  - Splitter: Provides more detailed control over long text splitting
  - PodCast: A tool to help create `long text` and `multi-character` audio based on podcast scripts
  - From subtitle: Create `SSML` scripts from subtitle files
- Voices (Speakers):
  - Builder: Create voices; currently supports creating voices from ChatTTS seeds or using reference audio to create `reference voices`
  - Test Voice: Test uploaded voice files
  - ChatTTS: Tools for debugging ChatTTS voices
    - Draw Cards: Create random voices using random seeds
    - Fusion: Merge voices created by different seeds
- ASR:
  - Whisper: Use the Whisper model for ASR
  - SenseVoice: WIP
- Tools: Various useful tools
  - Post Process: Post-processing tools for `editing`, `adjusting`, and `enhancing` audio

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

#### How to link to SillyTavern?

You can easily connect ChatTTS-Forge to your SillyTavern setup using the `/v1/xtts_v2` series of APIs.

Here's a simple configuration guide:

1. Open the Plugin Extension menu.
2. Open the `TTS` plugin configuration section.
3. Switch the `TTS Provider` to `XTTSv2`.
4. Check `Enabled`.
5. Select/Configure `Voice`.
6. **[Important]** Set the `Provider Endpoint` to `http://localhost:7870/v1/xtts_v2`.

![sillytavern_tts](./docs/sillytavern_tts.png)

## demo

### Styles Control

<details>
<summary>input</summary>

```xml
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        下面是一个 ChatTTS 用于合成多角色多情感的有声书示例[lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        黛玉冷笑道：[lbreak]
    </voice>
    <voice spk="female2" seed="42" style="angry">
        我说呢 [uv_break] ，亏了绊住，不然，早就飞起来了[lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        宝玉道：[lbreak]
    </voice>
    <voice spk="Alice" seed="42" style="unfriendly">
        “只许和你玩 [uv_break] ，替你解闷。不过偶然到他那里，就说这些闲话。”[lbreak]
    </voice>
    <voice spk="female2" seed="42" style="angry">
        “好没意思的话！[uv_break] 去不去，关我什么事儿？ 又没叫你替我解闷儿 [uv_break]，还许你不理我呢” [lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        说着，便赌气回房去了 [lbreak]
    </voice>
</speak>
```

</details>

<details>
<summary>output</summary>
  
[多角色.webm](https://github.com/lenML/ChatTTS-Forge/assets/37396659/82d91409-ad71-42ac-a4cd-d9c9340e3a07)

</details>

### Long Text

<details>
<summary>input</summary>

中华美食，作为世界饮食文化的瑰宝，以其丰富的种类、独特的风味和精湛的烹饪技艺而闻名于世。中国地大物博，各地区的饮食习惯和烹饪方法各具特色，形成了独树一帜的美食体系。从北方的京鲁菜、东北菜，到南方的粤菜、闽菜，无不展现出中华美食的多样性。

在中华美食的世界里，五味调和，色香味俱全。无论是辣味浓郁的川菜，还是清淡鲜美的淮扬菜，都能够满足不同人的口味需求。除了味道上的独特，中华美食还注重色彩的搭配和形态的美感，让每一道菜品不仅是味觉的享受，更是一场视觉的盛宴。

中华美食不仅仅是食物，更是一种文化的传承。每一道菜背后都有着深厚的历史背景和文化故事。比如，北京的烤鸭，代表着皇家气派；而西安的羊肉泡馍，则体现了浓郁的地方风情。中华美食的精髓在于它追求的“天人合一”，讲究食材的自然性和烹饪过程中的和谐。

总之，中华美食博大精深，其丰富的口感和多样的烹饪技艺，构成了一个充满魅力和无限可能的美食世界。无论你来自哪里，都会被这独特的美食文化所吸引和感动。

</details>

<details>
<summary>output</summary>

[long_text_demo.webm](https://github.com/lenML/ChatTTS-Forge/assets/37396659/fe18b0f1-a85f-4255-8e25-3c953480b881)

</details>

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

## Roadmap

### Model Supports

#### TTS

| Model      | Stream Mode    | vocie clone | training | support prompt | ready progress          |
| ---------- | -------------- | ----------- | -------- | -------------- | ----------------------- |
| ChatTTS    | token level    | ✅          | ❓       | ❓             | ✅                      |
| FishSpeech | sentence level | ✅          | ❓       | ❓             | ✅ (SFT version WIP 🚧) |
| CosyVoice  | sentence level | ✅          | ❓       | ✅             | ✅                      |
| FireRedTTS  | sentence level | ✅          | ❓       | ✅             | ✅                      |
| GPTSoVits  | sentence level | ✅          | ❓       | ❓             | 🚧                      |

#### ASR

| Model      | Streaming | training | mulit lang | ready progress |
| ---------- | --------- | -------- | ---------- | -------------- |
| Whisper    | ✅        | ❓       | ✅         | ✅             |
| SenseVoice | ✅        | ❓       | ✅         | 🚧             |

#### Voice Clone

| Model     | ready progress |
| --------- | -------------- |
| OpenVoice | ✅             |
| RVC       | 🚧             |

#### Enhancer

| Model           | ready progress |
| --------------- | -------------- |
| ResembleEnhance | ✅             |

## Model Download

Since Forge primarily focuses on API functionality development, automatic download logic has not yet been implemented. To download models, you need to manually invoke the download scripts, which can be found in the `./scripts` directory.

### Download Script

| Function    | Model          | Download Command                                            |
|-------------|----------------|-----------------------------------------------------------|
| **TTS**     | ChatTTS       | `python -m scripts.dl_chattts --source huggingface`     |
|             | FishSpeech    | `python -m scripts.downloader.fish_speech_1_2sft --source huggingface` |
|             | CosyVoice     | `python -m scripts.downloader.dl_cosyvoice_instruct --source huggingface` |
|             | FireRedTTS    | `python -m scripts.downloader.fire_red_tts --source huggingface` |
| **ASR**     | Whisper       | `python -m scripts.downloader.faster_whisper --source huggingface` |
| **CV**      | OpenVoice     | `python -m scripts.downloader.open_voice --source huggingface` |
| **Enhancer**| Enhancer Model | `python -m scripts.dl_enhance --source huggingface`     |

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
