[cn](./README.md) | [en](./README.en.md) | [Discord Server](https://discord.gg/9XnXUhAy3t)

# 🍦 ChatTTS-Forge

ChatTTS-Forge is a project developed around the TTS generation model ChatTTS, implementing an API Server and a Gradio-based WebUI.

![banner](./docs/banner.png)

You can experience and deploy ChatTTS-Forge through the following methods:

| -                        | Description                             | Link                                                                                                                                                                |
| ------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Online Demo**          | Deployed on HuggingFace                 | [HuggingFace Spaces](https://huggingface.co/spaces/lenML/ChatTTS-Forge)                                                                                             |
| **One-Click Start**      | Click the button to start Colab         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenML/ChatTTS-Forge/blob/main/colab.en.ipynb) |
| **Container Deployment** | See the docker section                  | [Docker](#docker)                                                                                                                                                   |
| **Local Deployment**     | See the environment preparation section | [Local Deployment](#InstallationandRunning)                                                                                                                         |

## 1. <a name='INDEX'></a>INDEX

<!-- vscode-markdown-toc -->

- 1. [INDEX](#INDEX)
- 2. [GPU Memory Requirements](#GPUMemoryRequirements)
  - 2.1. [Model Loading Memory Requirements](#ModelLoadingMemoryRequirements)
  - 2.2. [Batch Size Memory Requirements](#BatchSizeMemoryRequirements)
- 3. [ Installation and Running](#InstallationandRunning)
     - 3.1. [WebUI Features](#WebUIFeatures)
  - 3.1. [`launch.py`: API Server](#launch.py:APIServer)
    - 3.1.1. [How to link to SillyTavern?](#HowtolinktoSillyTavern)
- 4. [demo](#demo)
  - 4.1. [风格化控制](#)
  - 4.2. [长文本生成](#-1)
- 5. [Docker](#Docker)
  - 5.1. [Image](#Image)
  - 5.2. [Manual build](#Manualbuild)
- 6. [Roadmap](#Roadmap)
- 7. [FAQ](#FAQ)
  - 7.1. [What are Prompt1 and Prompt2?](#WhatarePrompt1andPrompt2)
  - 7.2. [What is Prefix?](#WhatisPrefix)
  - 7.3. [What is the difference in Style with `_p`?](#WhatisthedifferenceinStylewith_p)
  - 7.4. [Why is it slow when `--compile` is enabled?](#Whyisitslowwhen--compileisenabled)
  - 7.5. [7.5. Why is Colab very slow with only 2 it/s?](#WhyisColabveryslowwithonly2its)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## 2. <a name='GPUMemoryRequirements'></a>GPU Memory Requirements

### 2.1. <a name='ModelLoadingMemoryRequirements'></a>Model Loading Memory Requirements

| Data Type | Load ChatTTS Model | Load Enhancer Model |
| --------- | ------------------ | ------------------- |
| float32   | 2GB                | 3GB                 |
| half      | 1GB                | 1.5GB               |

### 2.2. <a name='BatchSizeMemoryRequirements'></a>Batch Size Memory Requirements

| Data Type | Batch Size | Without Enhancer | With Enhancer |
| --------- | ---------- | ---------------- | ------------- |
| float32   | ≤ 4        | 2GB              | 4GB           |
| float32   | 8          | 8~10GB           | 8~14GB        |
| half      | ≤ 4        | 2GB              | 4GB           |
| half      | 8          | 2~6GB            | 4~8GB         |

**Notes:**

- For Batch Size ≤ 4, 2GB of memory is sufficient for inference.
- For Batch Size = 8, 8~14GB of memory is required.
- Half Batch Size means half of the above-mentioned Batch Size, and the memory requirements are also halved accordingly.

## 3. <a name='InstallationandRunning'></a> Installation and Running

1. Ensure that the [related dependencies](./docs/dependencies.md) are correctly installed.
2. Start the required services according to your needs.

- webui: `python webui.py`
- api: `python launch.py`

#### 3.1. <a name='WebUIFeatures'></a>WebUI Features

[Click here for a detailed introduction with images](./docs/webui_features.md)

- Native functions of ChatTTS model: Refiner/Generate
- Native Batch synthesis for efficient long text synthesis
- Style control
- SSML
  - Editor: Simple SSML editing, used in conjunction with other features
  - Splitter: Preprocessing for long text segmentation
  - Podcast: Support for creating and editing podcast scripts
- Speaker
  - Built-in voices: A variety of built-in speakers available
  - Speaker creator: Supports voice testing and creation of new speakers
  - Embedding: Supports uploading speaker embeddings to reuse saved speakers
  - Speaker merge: Supports merging speakers and fine-tuning
- Prompt Slot
- Text normalization
- Audio quality enhancement:
  - Enhance: Improves output quality
  - Denoise: Removes noise
- Experimental features:
  - fintune
    - speaker embedding
    - [WIP] GPT lora
    - [WIP] AE
  - [WIP] ASR
  - [WIP] Inpainting

### 3.1. <a name='launch.py:APIServer'></a>`launch.py`: API Server

Launch.py is the startup script for ChatTTS-Forge, used to configure and launch the API server.

Once the `launch.py` script has started successfully, you can check if the API is enabled at `/docs`.

[Detailed API documentation](./docs/api.md)

#### 3.1.1. <a name='HowtolinktoSillyTavern'></a>How to link to SillyTavern?

Through the `/v1/xtts_v2` series API, you can easily connect ChatTTS-Forge to your SillyTavern.

Here's a simple configuration guide:

1. Open the plugin extension.
2. Open the `TTS` plugin configuration section.
3. Switch `TTS Provider` to `XTTSv2`.
4. Check `Enabled`.
5. Select/configure `Voice`.
6. **[Key Step]** Set the `Provider Endpoint` to `http://localhost:7870/v1/xtts_v2`.

![sillytavern_tts](./docs/sillytavern_tts.png)

## 4. <a name='demo'></a>demo

### 4.1. <a name=''></a>风格化控制

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

<details open>
<summary>output</summary>
  
[多角色.webm](https://github.com/lenML/ChatTTS-Forge/assets/37396659/82d91409-ad71-42ac-a4cd-d9c9340e3a07)

</details>

### 4.2. <a name='-1'></a>长文本生成

<details>
<summary>input</summary>

中华美食，作为世界饮食文化的瑰宝，以其丰富的种类、独特的风味和精湛的烹饪技艺而闻名于世。中国地大物博，各地区的饮食习惯和烹饪方法各具特色，形成了独树一帜的美食体系。从北方的京鲁菜、东北菜，到南方的粤菜、闽菜，无不展现出中华美食的多样性。

在中华美食的世界里，五味调和，色香味俱全。无论是辣味浓郁的川菜，还是清淡鲜美的淮扬菜，都能够满足不同人的口味需求。除了味道上的独特，中华美食还注重色彩的搭配和形态的美感，让每一道菜品不仅是味觉的享受，更是一场视觉的盛宴。

中华美食不仅仅是食物，更是一种文化的传承。每一道菜背后都有着深厚的历史背景和文化故事。比如，北京的烤鸭，代表着皇家气派；而西安的羊肉泡馍，则体现了浓郁的地方风情。中华美食的精髓在于它追求的“天人合一”，讲究食材的自然性和烹饪过程中的和谐。

总之，中华美食博大精深，其丰富的口感和多样的烹饪技艺，构成了一个充满魅力和无限可能的美食世界。无论你来自哪里，都会被这独特的美食文化所吸引和感动。

</details>

<details open>
<summary>output</summary>

[long_text_demo.webm](https://github.com/lenML/ChatTTS-Forge/assets/37396659/fe18b0f1-a85f-4255-8e25-3c953480b881)

</details>

## 5. <a name='Docker'></a>Docker

### 5.1. <a name='Image'></a>Image

WIP

### 5.2. <a name='Manualbuild'></a>Manual build

download models

```bash
python -m scripts.download_models --source huggingface
```

- webui: `docker-compose -f ./docker-compose.webui.yml up -d`
- api: `docker-compose -f ./docker-compose.api.yml up -d`

Environment variable configuration

- webui: [.env.webui](./.env.webui)
- api: [.env.api](./.env.api)

## 6. <a name='Roadmap'></a>Roadmap

WIP

## 7. <a name='FAQ'></a>FAQ

### 7.1. <a name='WhatarePrompt1andPrompt2'></a>What are Prompt1 and Prompt2?

Prompt1 and Prompt2 are system prompts with different insertion points. The current model is very sensitive to the first [Stts] token, hence the need for two prompts.

- Prompt1 is inserted before the first [Stts].
- Prompt2 is inserted after the first [Stts].

### 7.2. <a name='WhatisPrefix'></a>What is Prefix?

The prefix is primarily used to control the model's generation capabilities, similar to the refine prompt in the official examples. This prefix should only contain special non-lexical tokens, such as `[laugh_0]`, `[oral_0]`, `[speed_0]`, `[break_0]`, etc.

### 7.3. <a name='WhatisthedifferenceinStylewith_p'></a>What is the difference in Style with `_p`?

Styles with `_p` use both prompt and prefix, while those without `_p` use only the prefix.

### 7.4. <a name='Whyisitslowwhen--compileisenabled'></a>Why is it slow when `--compile` is enabled?

Due to the lack of inference padding, any change in the inference shape may trigger torch to compile.

> It is currently not recommended to enable this.

### 7.5. <a name='WhyisColabveryslowwithonly2its'></a>7.5. Why is Colab very slow with only 2 it/s?

Make sure you are using a GPU instead of a CPU.

- Click on the menu bar **[Edit]**
- Click **[Notebook settings]**
- Select **[Hardware accelerator]** => T4 GPU

# Contributing

To contribute, clone the repository, make your changes, commit and push to your clone, and submit a pull request.

# References

- ChatTTS: https://github.com/2noise/ChatTTS
- PaddleSpeech: https://github.com/PaddlePaddle/PaddleSpeech
- resemble-enhance: https://github.com/resemble-ai/resemble-enhance
- 默认说话人: https://github.com/2noise/ChatTTS/issues/238
