[cn](./README.md) | [en](./README.en.md)

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
- 2. [Features](#Features)
- 3. [Interface](#Interface)
- 4. [ Installation and Running](#InstallationandRunning)
  - 4.1. [ `webui.py`: WebUI](#webui.py:WebUI)
    - 4.1.1. [WebUI Features](#WebUIFeatures)
  - 4.2. [`launch.py`: API Server](#launch.py:APIServer)
- 5. [Benchmark](#Benchmark)
- 6. [demo](#demo)
  - 6.1. [风格化控制](#)
  - 6.2. [长文本生成](#-1)
- 7. [SSML](#SSML)
- 8. [Speaking style](#Speakingstyle)
- 9. [Docker](#Docker)
  - 9.1. [镜像](#-1)
  - 9.2. [手动 build](#build)
- 10. [Roadmap](#Roadmap)
- 11. [FAQ](#FAQ)
  - 11.1. [What are Prompt1 and Prompt2?](#WhatarePrompt1andPrompt2)
  - 11.2. [What is Prefix?](#WhatisPrefix)
  - 11.3. [What is the difference in Style with `_p`?](#WhatisthedifferenceinStylewith_p)
  - 11.4. [Why is it slow when `--compile` is enabled?](#Whyisitslowwhen--compileisenabled)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## 2. <a name='Features'></a>Features

- **Comprehensive API Services**: Provides API access to all functionalities for easy integration.
- **Ultra-long Text Generation**: Supports generating texts longer than 1000 characters while maintaining consistency.
- **Style Management**: Reuse speaking styles by name or ID, with 32 built-in styles.
- **Speaker Management**: Efficiently reuse speakers by name or ID.
- **Style Prompt Injection**: Flexibly adjust output styles by injecting prompts.
- **Batch Generation**: Supports automatic bucketing and batch generation.
- **SSML-like Support**: Create rich audio long texts using SSML-like syntax.
- **Independent Refine API**: Provides a separate refine debugging interface to improve debugging efficiency.
- **OpenAI-style API**: Provides a speech generation interface similar to OpenAI's `/v1/audio/speech`.
- **Google-style API**: Provides a text synthesis interface similar to Google's `/v1/text:synthesize`.
- **User-friendly Debugging GUI**: An independent playground from Gradio to simplify the debugging process.
- **Text Normalization**:
  - **Markdown**: Automatically detects and processes markdown formatted texts.
  - **Number Transcription**: Automatically converts numbers to text recognizable by the model.
  - **Emoji Adaptation**: Automatically translates emojis into readable text.
  - **Tokenizer-based**: Preprocesses texts based on tokenizer, covering all unsupported character ranges of the model.
  - **Chinese-English Recognition**: Adapts to English environments.
- **Audio Quality Enhancement**: Inherits audio quality enhancement and noise reduction models to improve output quality.
- **Speaker Import and Export**: Supports importing and exporting speakers for easy customization.
- **Speaker Merging**: Supports merging speakers and fine-tuning them.

## 3. <a name='Interface'></a>Interface

<table>
  <tr>
    <th>Project</th>
    <th>Description</th>
    <th>Deployment or Usage Method</th>
    <th>Image</th>
  </tr>
  <tr>
    <td rowspan="2">API</td>
    <td>Provides various forms of text-to-speech interfaces. Visit <code>http://localhost:8000/docs</code> for detailed information after deployment.</td>
    <td>Run <code>python launch.py</code></td>
    <td rowspan="2"><img src="./docs/api.png" alt="API Documentation"><br><img src="./docs/playground.png" alt="Playground"></td>
  </tr>
  <tr>
    <td>Includes a Playground front-end page independent of Python code and Gradio, facilitating API debugging.</td>
    <td>Visit <code>http://localhost:8000/playground/index.html</code> after deployment</td>
  </tr>
  <tr>
    <td>WebUI</td>
    <td>In some scenarios (such as HuggingFace/Colab), WebUI is needed. Here is a simple implementation. Please note that WebUI does not support write operations to any local files.</td>
    <td>Run <code>python webui.py</code></td>
    <td><img src="./docs/webui.png" alt="WebUI"></td>
  </tr>
</table>

## 4. <a name='InstallationandRunning'></a> Installation and Running

1. Ensure that the [related dependencies](./docs/dependencies.md) are correctly installed.
2. Start the required services according to your needs. The specific startup parameters are as follows.

### 4.1. <a name='webui.py:WebUI'></a> `webui.py`: WebUI

WebUI.py is a script used to configure and start the Gradio Web UI interface.

All parameters:

| Parameter              | Type   | Default     | Description                                                                |
| ---------------------- | ------ | ----------- | -------------------------------------------------------------------------- |
| `--server_name`        | `str`  | `"0.0.0.0"` | Server host address                                                        |
| `--server_port`        | `int`  | `7860`      | Server port                                                                |
| `--share`              | `bool` | `False`     | Enable share mode, allowing external access                                |
| `--debug`              | `bool` | `False`     | Enable debug mode                                                          |
| `--compile`            | `bool` | `False`     | Enable model compilation                                                   |
| `--auth`               | `str`  | `None`      | Username and password for authentication in the format `username:password` |
| `--half`               | `bool` | `False`     | Enable f16 half-precision inference                                        |
| `--off_tqdm`           | `bool` | `False`     | Disable tqdm progress bar                                                  |
| `--tts_max_len`        | `int`  | `1000`      | Maximum text length for TTS (Text-to-Speech)                               |
| `--ssml_max_len`       | `int`  | `2000`      | Maximum text length for SSML (Speech Synthesis Markup Language)            |
| `--max_batch_size`     | `int`  | `8`         | Maximum batch size for TTS                                                 |
| `--device_id`          | `str`  | `None`      | Specify the GPU device_id to use                                           |
| `--use_cpu`            | `str`  | `None`      | Currently selectable value is `"all"`                                      |
| `--webui_experimental` | `bool` | `False`     | Enable experimental features (incomplete features)                         |
| `--language`           | `str`  | `zh-CN`     | Set webui localization                                                     |

> Enabling `--half` can significantly reduce memory usage. If the batch size is greater than 8, it is recommended to enable half.

> Since `MKL FFT doesn't support tensors of type: Half`, `--half` and `--use_cpu="all"` cannot be used simultaneously.

#### 4.1.1. <a name='WebUIFeatures'></a>WebUI Features

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
  - [WIP] ASR
  - [WIP] Inpainting

### 4.2. <a name='launch.py:APIServer'></a>`launch.py`: API Server

Launch.py is the startup script for ChatTTS-Forge, used to configure and launch the API server.

All parameters:

| Parameter         | Type   | Default     | Description                                                             |
| ----------------- | ------ | ----------- | ----------------------------------------------------------------------- |
| `--host`          | `str`  | `"0.0.0.0"` | Server host address                                                     |
| `--port`          | `int`  | `8000`      | Server port                                                             |
| `--reload`        | `bool` | `False`     | Enable auto-reload (for development)                                    |
| `--compile`       | `bool` | `False`     | Enable model compilation                                                |
| `--lru_size`      | `int`  | `64`        | Set the size of the request cache pool; set to 0 to disable `lru_cache` |
| `--cors_origin`   | `str`  | `"*"`       | Allowed CORS origins; use `*` to allow all origins                      |
| `--no_playground` | `bool` | `False`     | Disable playground entry                                                |
| `--no_docs`       | `bool` | `False`     | Disable docs entry                                                      |
| `--half`          | `bool` | `False`     | Enable f16 half-precision inference                                     |
| `--off_tqdm`      | `bool` | `False`     | Disable tqdm progress bar                                               |
| `--exclude`       | `str`  | `""`        | Exclude unnecessary APIs                                                |
| `--device_id`     | `str`  | `None`      | Specify GPU device ID                                                   |
| `--use_cpu`       | `str`  | `None`      | Current optional value is `"all"`                                       |

Once the `launch.py` script has started successfully, you can check if the API is enabled at `/docs`.

[Detailed API documentation](./docs/api.md)

## 5. <a name='Benchmark'></a>Benchmark

> You can reproduce this using `./tests/benchmark/tts_benchmark.py`

Test Platform

- GPU: `GeForce RTX 2080 Ti`
- CPU: `3.4Hz 24-core`

The results for a batch size of 8 are as follows. For the full scan, see `performance_results.csv`.

| Batch size | Use decoder | Half precision | Compile model | Use CPU | GPU Memory | Duration | RTF  |
| ---------- | ----------- | -------------- | ------------- | ------- | ---------- | -------- | ---- |
| 8          | ✅          | ❌             | ✅            | ❌      | 1.72       | 36.78    | 0.22 |
| 8          | ✅          | ✅             | ✅            | ❌      | 0.89       | 39.34    | 0.24 |
| 8          | ❌          | ❌             | ✅            | ❌      | 1.72       | 36.78    | 0.23 |
| 8          | ❌          | ✅             | ✅            | ❌      | 0.90       | 39.34    | 0.24 |
| 8          | ❌          | ❌             | ❌            | ❌      | 1.70       | 36.78    | 0.29 |
| 8          | ✅          | ❌             | ❌            | ❌      | 1.72       | 36.78    | 0.29 |
| 8          | ❌          | ✅             | ❌            | ❌      | 1.02       | 35.75    | 0.40 |
| 8          | ✅          | ✅             | ❌            | ❌      | 0.95       | 35.75    | 0.40 |
| 8          | ❌          | ❌             | ❌            | ✅      | N/A        | 49.92    | 0.58 |
| 8          | ❌          | ❌             | ✅            | ✅      | N/A        | 49.92    | 0.58 |
| 8          | ✅          | ❌             | ✅            | ✅      | N/A        | 49.92    | 0.58 |
| 8          | ✅          | ❌             | ❌            | ✅      | N/A        | 49.92    | 0.60 |
| 8          | ❌          | ✅             | ❌            | ✅      | N/A        | N/A      | N/A  |
| 8          | ❌          | ✅             | ✅            | ✅      | N/A        | N/A      | N/A  |
| 8          | ✅          | ✅             | ❌            | ✅      | N/A        | N/A      | N/A  |
| 8          | ✅          | ✅             | ✅            | ✅      | N/A        | N/A      | N/A  |

## 6. <a name='demo'></a>demo

### 6.1. <a name=''></a>风格化控制

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

### 6.2. <a name='-1'></a>长文本生成

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

## 7. <a name='SSML'></a>SSML

[SSML readme](./docs/SSML.md)

## 8. <a name='Speakingstyle'></a>Speaking style

[style readme](./docs/sytles.md)

## 9. <a name='Docker'></a>Docker

### 9.1. <a name='-1'></a>Image

WIP

### 9.2. <a name='build'></a>Manual build

download models

```bash
python -m scripts.download_models --source huggingface
```

- webui: `docker-compose -f ./docker-compose.webui.yml up -d`
- api: `docker-compose -f ./docker-compose.api.yml up -d`

Environment variable configuration

- webui: [.env.webui](./.env.webui)
- api: [.env.api](./.env.api)

## 10. <a name='Roadmap'></a>Roadmap

WIP

## 11. <a name='FAQ'></a>FAQ

### 11.1. <a name='WhatarePrompt1andPrompt2'></a>What are Prompt1 and Prompt2?

Prompt1 and Prompt2 are system prompts with different insertion points. The current model is very sensitive to the first [Stts] token, hence the need for two prompts.

- Prompt1 is inserted before the first [Stts].
- Prompt2 is inserted after the first [Stts].

### 11.2. <a name='WhatisPrefix'></a>What is Prefix?

The prefix is primarily used to control the model's generation capabilities, similar to the refine prompt in the official examples. This prefix should only contain special non-lexical tokens, such as `[laugh_0]`, `[oral_0]`, `[speed_0]`, `[break_0]`, etc.

### 11.3. <a name='WhatisthedifferenceinStylewith_p'></a>What is the difference in Style with `_p`?

Styles with `_p` use both prompt and prefix, while those without `_p` use only the prefix.

### 11.4. <a name='Whyisitslowwhen--compileisenabled'></a>Why is it slow when `--compile` is enabled?

Due to the lack of inference padding, any change in the inference shape may trigger torch to compile.

> It is currently not recommended to enable this.

# References

- ChatTTS: https://github.com/2noise/ChatTTS
- PaddleSpeech: https://github.com/PaddlePaddle/PaddleSpeech
- resemble-enhance: https://github.com/resemble-ai/resemble-enhance
- 默认说话人: https://github.com/2noise/ChatTTS/issues/238
