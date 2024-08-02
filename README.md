[cn](./README.md) | [en](./README.en.md) | [Discord Server](https://discord.gg/9XnXUhAy3t)

# 🍦 ChatTTS-Forge

ChatTTS-Forge 是一个围绕 TTS 生成模型开发的项目，实现了 API Server 和 基于 Gradio 的 WebUI。

![banner](./docs/banner.png)

你可以通过以下几种方式体验和部署 ChatTTS-Forge：

| -            | 描述                     | 链接                                                                                                                                                             |
| ------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **在线体验** | 部署于 HuggingFace 中    | [HuggingFace Spaces](https://huggingface.co/spaces/lenML/ChatTTS-Forge)                                                                                          |
| **一键启动** | 点击按钮，一键启动 Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenML/ChatTTS-Forge/blob/main/colab.ipynb) |
| **容器部署** | 查看 docker 部分         | [Docker](#docker)                                                                                                                                                |
| **本地部署** | 查看环境准备部分         | [本地部署](#InstallationandRunning)                                                                                                                              |

## 1. <a name='INDEX'></a>INDEX

<!-- vscode-markdown-toc -->

- 1. [INDEX](#INDEX)
- 2. [GPU 显存要求](#GPU)
  - 2.1. [加载模型显存要求](#)
  - 2.2. [Batch Size 显存要求](#BatchSize)
- 3. [ Installation and Running](#InstallationandRunning)
  - 3.1. [webui features](#webuifeatures)
  - 3.2. [`launch.py`: API Server](#launch.py:APIServer)
    - 3.2.1. [How to link to SillyTavern?](#HowtolinktoSillyTavern)
- 4. [demo](#demo)
  - 4.1. [风格化控制](#-1)
  - 4.2. [长文本生成](#-1)
- 5. [Docker](#Docker)
  - 5.1. [镜像](#-1)
  - 5.2. [手动 build](#build)
- 6. [Roadmap](#Roadmap)
- 7. [FAQ](#FAQ)
  - 7.1. [什么是 Prompt1 和 Prompt2？](#Prompt1Prompt2)
  - 7.2. [什么是 Prefix？](#Prefix)
  - 7.3. [Style 中 `_p` 的区别是什么？](#Style_p)
  - 7.4. [为什么开启了 `--compile` 很慢？](#--compile)
  - 7.5. [为什么 colab 里面非常慢只有 2 it/s ？](#colab2its)
- 8. [离线整合包](#-1)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## 2. GPU 显存要求

### 2.1. 模型加载显存需求

| 精度   | ChatTTS 模型 | Enhancer 模型 |
| ------ | ------------ | ------------- |
| 全精度 | 2GB          | 3GB           |
| 半精度 | 1GB          | 1.5GB         |

注：半精度为默认设置，全精度可通过 `--no_half` 参数启用。

### 2.2. 推理过程显存需求

| 精度   | Batch Size | 不使用 Enhancer | 使用 Enhancer |
| ------ | ---------- | --------------- | ------------- |
| 全精度 | ≤ 4        | 2GB             | 4GB           |
| 全精度 | 8          | 4-10GB          | 6-14GB        |
| 半精度 | ≤ 4        | 1GB             | 2GB           |
| 半精度 | 8          | 2-6GB           | 4-8GB         |

注意事项：

1. 显存需求与上下文长度相关，因此呈现为一个范围。
2. 半精度（默认）的显存需求约为全精度的一半。
3. 对于 Batch Size ≤ 4，4GB 显存通常足够进行推理。
4. Batch Size 为 8 时，可能需要 6-14GB 显存，具体取决于精度和是否使用 Enhancer。

## 3. <a name='InstallationandRunning'></a> Installation and Running

1. 确保 [相关依赖](./docs/dependencies.md) 已经正确安装，
2. 根据你的需求启动需要的服务。

- webui: `python webui.py`
- api: `python launch.py`

### 3.1. <a name='webuifeatures'></a>webui features

[点我看详细图文介绍](./docs/webui_features.md)

- ChatTTS 模型原生功能 Refiner/Generate
- 原生 Batch 合成，高效合成超长文本
- Style control
- SSML
  - Editor: 简单的 SSML 编辑，配合其他功能使用
  - Spliter：超长文本分割预处理
  - Podcast: 支持创建编辑播客脚本
- Speaker
  - 内置音色：内置众多 speaker 可以使用
  - speaker creator: 支持试音抽卡，创建 speaker
  - embdding: 支持 speaker embdding 上传，可以复用保存下来的 speaker
  - speaker merge: 支持合并说话人，微调 speaker
- Prompt Slot
- Text Normalize
- 音质增强：
  - enhance: 音质增强提高输出质量
  - denoise: 去除噪音
- Experimental 实验功能
  - fintune
    - speaker embedding
    - [WIP] GPT lora
    - [WIP] AE
  - [WIP] ASR
  - [WIP] Inpainting

### 3.2. <a name='launch.py:APIServer'></a>`launch.py`: API Server

某些情况，你并不需要 webui，那么可以使用这个脚本启动单纯的 api 服务。

launch.py 脚本启动成功后，你可以在 `/docs` 下检查 api 是否开启。

[详细 API 文档](./docs/api.md)

#### 3.2.1. <a name='HowtolinktoSillyTavern'></a>How to link to SillyTavern?

通过 `/v1/xtts_v2` 系列 api，你可以方便的将 ChatTTS-Forge 连接到你的 SillyTavern 中。

下面是一个简单的配置指南:

1. 点开 插件拓展
2. 点开 `TTS` 插件配置部分
3. 切换 `TTS Provider` 为 `XTTSv2`
4. 勾选 `Enabled`
5. 选择/配置 `Voice`
6. **[关键]** 设置 `Provider Endpoint` 到 `http://localhost:7870/v1/xtts_v2`

![sillytavern_tts](./docs/sillytavern_tts.png)

## 4. <a name='demo'></a>demo

### 4.1. <a name='-1'></a>风格化控制

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

### 5.1. <a name='-1'></a>镜像

WIP 开发中

### 5.2. <a name='build'></a>手动 build

下载模型: `python -m scripts.download_models --source modelscope`

- webui: `docker-compose -f ./docker-compose.webui.yml up -d`
- api: `docker-compose -f ./docker-compose.api.yml up -d`

环境变量配置

- webui: [.env.webui](./.env.webui)
- api: [.env.api](./.env.api)

## 6. <a name='Roadmap'></a>Roadmap

### Model Supports

#### TTS

| 模型名称   | 流式级别 | 支持复刻 | 支持训练 | 支持 prompt | 实现情况               |
| ---------- | -------- | -------- | -------- | ----------- | ---------------------- |
| ChatTTS    | token 级 | ✅       | ❓       | ❓          | ✅                     |
| FishSpeech | 句子级   | ✅       | ❓       | ❓          | ✅ (SFT 版本开发中 🚧) |
| CosyVoice  | 句子级   | ✅       | ❓       | ✅          | ✅                     |

#### ASR

| 模型名称   | 流式识别 | 支持训练 | 支持多语言 | 实现情况 |
| ---------- | -------- | -------- | ---------- | -------- |
| Whisper    | ✅       | ❓       | ✅         | ✅       |
| SenseVoice | ✅       | ❓       | ✅         | 🚧       |

#### Voice Clone

| 模型名称  | 实现情况 |
| --------- | -------- |
| OpenVoice | 🚧       |
| RVC       | 🚧       |

#### Enhancer

| 模型名称        | 实现情况 |
| --------------- | -------- |
| ResembleEnhance | ✅       |

## 7. <a name='FAQ'></a>FAQ

### 7.1. <a name='Prompt1Prompt2'></a>什么是 Prompt1 和 Prompt2？

Prompt1 和 Prompt2 都是系统提示（system prompt），区别在于插入点不同。因为测试发现当前模型对第一个 [Stts] token 非常敏感，所以需要两个提示。

- Prompt1 插入到第一个 [Stts] 之前
- Prompt2 插入到第一个 [Stts] 之后

### 7.2. <a name='Prefix'></a>什么是 Prefix？

Prefix 主要用于控制模型的生成能力，类似于官方示例中的 refine prompt。这个 prefix 中应该只包含特殊的非语素 token，如 `[laugh_0]`、`[oral_0]`、`[speed_0]`、`[break_0]` 等。

### 7.3. <a name='Style_p'></a>Style 中 `_p` 的区别是什么？

Style 中带有 `_p` 的使用了 prompt + prefix，而不带 `_p` 的则只使用 prefix。

### 7.4. <a name='--compile'></a>为什么开启了 `--compile` 很慢？

由于还未实现推理 padding 所以如果每次推理 shape 改变都可能触发 torch 进行 compile

> 暂时不建议开启

### 7.5. <a name='colab2its'></a>为什么 colab 里面非常慢只有 2 it/s ？

请确保使用 gpu 而非 cpu。

- 点击菜单栏 【修改】
- 点击 【笔记本设置】
- 选择 【硬件加速器】 => T4 GPU

## 8. <a name='-1'></a>离线整合包

感谢 @Phrixus2023 提供的整合包：
https://pan.baidu.com/s/1Q1vQV5Gs0VhU5J76dZBK4Q?pwd=d7xu

相关讨论：
https://github.com/lenML/ChatTTS-Forge/discussions/65

# Documents

在这里可以找到 [更多文档](./docs/readme.md)

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
