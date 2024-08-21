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
- 2. [ Installation and Running](#InstallationandRunning)
  - 2.1. [webui features](#webuifeatures)
  - 2.2. [`launch.py`: API Server](#launch.py:APIServer)
    - 2.2.1. [How to link to SillyTavern?](#HowtolinktoSillyTavern)
- 3. [demo](#demo)
  - 3.1. [风格化控制](#)
  - 3.2. [长文本生成](#-1)
- 4. [Docker](#Docker)
  - 4.1. [镜像](#-1)
  - 4.2. [手动 build](#build)
- 5. [Roadmap](#Roadmap)
  - 5.1. [Model Supports](#ModelSupports)
    - 5.1.1. [TTS](#TTS)
    - 5.1.2. [ASR](#ASR)
    - 5.1.3. [Voice Clone](#VoiceClone)
    - 5.1.4. [Enhancer](#Enhancer)
    - 5.1.5. [模型下载](#-1)
- 6. [FAQ](#FAQ)
  - 6.1. [什么是 Prompt1 和 Prompt2？](#Prompt1Prompt2)
  - 6.2. [什么是 Prefix？](#Prefix)
  - 6.3. [Style 中 `_p` 的区别是什么？](#Style_p)
  - 6.4. [为什么开启了 `--compile` 很慢？](#--compile)
  - 6.5. [为什么 colab 里面非常慢只有 2 it/s ？](#colab2its)
- 7. [离线整合包](#-1)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## 2. <a name='InstallationandRunning'></a> Installation and Running

首先，确保 [相关依赖](./docs/dependencies.md) 已经正确安装

启动：

```
python webui.py
```

### 2.1. <a name='webuifeatures'></a>webui features

[点我看详细图文介绍](./docs/webui_features.md)

- TTS: tts 模型的功能
  - Speaker Switch: 可以切换音色
    - 内置音色： 内置多个音色可使用， `27 ChatTTS` / `7 CosyVoice` 音色 + `1 参考音色`
    - 音色上传： 支持上传自定义音色文件，并实时推理
    - 参考音色： 支持上传参考音频/文本，直接使用参考音频进行 `tts` 推理
  - Style： 风格控制内置多种风格控制
  - Long Text： 支持超长文本推理，自动分割文本
    - Batch Size： 可设置 `Batch size` ，对于支持 `batch` 推理的模型长文本推理速度更快
  - Refiner: 支持 `ChatTTS` 原生文本 `refiner` ，同时支持无限长文本
  - 分割器： 可调整分割器配置，控制分割器 `eos` 和 `分割阈值`
  - 调节器： 支持对 `速度/音调/音量` 调整，并增加实用的 `响度均衡` 功能
  - 人声增强： 支持使用 `Enhancer` 模型增强 `TTS` 输出结果，进一步提高输出质量
  - 生成历史： 支持保留最近三次生成结果，方便对比
  - 多模型： 支持多种 `TTS` 模型推理，包括 `ChatTTS` / `CosyVoice` / `FishSpeech` / `GPT-SoVITS` 等
- SSML: 类 XML 语法的高级 TTS 合成控制工具
  - 分割器： 在这里面可以更加细致的控制长文本分割结果
  - PodCast： 博客工具，帮助你根据博客脚本创建 `长文本`、`多角色` 音频
  - From subtitle： 从字幕文件创建 `SSML` 脚本
- 音色：
  - Builder： 创建音色，目前可以从 ChatTTS seed 创建音色、或者使用 Refrence Audio 创建 `参考音色`
  - Test Voice： 试音，上传音色文件，简单测试音色
  - ChatTTS: 针对 ChatTTS 音色的调试工具
    - 抽卡： 使用随机种子抽卡，创建随机音色
    - 融合： 融合不同种子创建的音色
- ASR:
  - Whisper: 使用 whisper 模型进行 asr
  - SenseVoice： WIP
- Tools： 一些实用的工具
  - Post Process: 后处理工具，可以在这里 `剪辑`、`调整`、`增强` 音频

### 2.2. <a name='launch.py:APIServer'></a>`launch.py`: API Server

某些情况，你并不需要 webui 或者需要更高的 api 吞吐，那么可以使用这个脚本启动单纯的 api 服务。

启动：

```
python launch.py
```

启动之后开启 `http://localhost:7870/docs` 可以查看开启了哪些 api 端点

更多帮助信息:

- 通过 `python launch.py -h` 查看脚本参数
- 查看 [API 文档](./docs/api.md)

#### 2.2.1. <a name='HowtolinktoSillyTavern'></a>How to link to SillyTavern?

通过 `/v1/xtts_v2` 系列 api，你可以方便的将 ChatTTS-Forge 连接到你的 SillyTavern 中。

下面是一个简单的配置指南:

1. 点开 插件拓展
2. 点开 `TTS` 插件配置部分
3. 切换 `TTS Provider` 为 `XTTSv2`
4. 勾选 `Enabled`
5. 选择/配置 `Voice`
6. **[关键]** 设置 `Provider Endpoint` 到 `http://localhost:7870/v1/xtts_v2`

![sillytavern_tts](./docs/sillytavern_tts.png)

## 3. <a name='demo'></a>demo

### 3.1. <a name=''></a>风格化控制

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

### 3.2. <a name='-1'></a>长文本生成

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

## 4. <a name='Docker'></a>Docker

### 4.1. <a name='-1'></a>镜像

WIP 开发中

### 4.2. <a name='build'></a>手动 build

下载模型: `python -m scripts.download_models --source modelscope`

> 此脚本将下载 `chat-tts` 和 `enhancer` 模型，如需下载其他模型，请看后续的 `模型下载` 介绍

- webui: `docker-compose -f ./docker-compose.webui.yml up -d`
- api: `docker-compose -f ./docker-compose.api.yml up -d`

环境变量配置

- webui: [.env.webui](./.env.webui)
- api: [.env.api](./.env.api)

## 5. <a name='Roadmap'></a>Roadmap

### 5.1. <a name='ModelSupports'></a>Model Supports

#### 5.1.1. <a name='TTS'></a>TTS

| 模型名称   | 流式级别 | 支持复刻 | 支持训练 | 支持 prompt | 实现情况               |
| ---------- | -------- | -------- | -------- | ----------- | ---------------------- |
| ChatTTS    | token 级 | ✅       | ❓       | ❓          | ✅                     |
| FishSpeech | 句子级   | ✅       | ❓       | ❓          | ✅ (SFT 版本开发中 🚧) |
| CosyVoice  | 句子级   | ✅       | ❓       | ✅          | ✅                     |
| GPTSoVits  | 句子级   | ✅       | ❓       | ❓          | 🚧                     |

#### 5.1.2. <a name='ASR'></a>ASR

| 模型名称   | 流式识别 | 支持训练 | 支持多语言 | 实现情况 |
| ---------- | -------- | -------- | ---------- | -------- |
| Whisper    | ✅       | ❓       | ✅         | ✅       |
| SenseVoice | ✅       | ❓       | ✅         | 🚧       |

#### 5.1.3. <a name='VoiceClone'></a>Voice Clone

| 模型名称  | 实现情况 |
| --------- | -------- |
| OpenVoice | ✅       |
| RVC       | 🚧       |

#### 5.1.4. <a name='Enhancer'></a>Enhancer

| 模型名称        | 实现情况 |
| --------------- | -------- |
| ResembleEnhance | ✅       |

#### 5.1.5. <a name='-1'></a>模型下载

由于 forge 主要是面向 api 功能开发，所以目前暂未实现自动下载逻辑，下载模型需手动调用下载脚本，具体脚本在 `./scripts` 目录下

下面列出一些下载脚本使用示例：

- TTS
  - 下载 ChatTTS： `python -m scripts.dl_chattts.py --source huggingface`
  - 下载 FishSpeech： `python -m scripts.downloader.fish_speech_1_2sft.py --source huggingface`
  - 下载 CosyVoice： `python -m scripts.downloader.dl_cosyvoice_instruct.py --source huggingface`
- ASR
  - 下载 Whisper： `python -m scripts.downloader.fish_speech_1_2sft.py --source huggingface`
- CV
  - OpenVoice: `python -m scripts.downloader.open_voice.py --source huggingface`
- Enhancer: `python -m scripts.dl_enhance.py --source huggingface`

> 其中若需要使用 model scope 下载模型，使用 `--source modelscope` 即可。
> 注：部分模型无法使用 model scope 下载，因为其中没有

## 6. <a name='FAQ'></a>FAQ

### 6.1. <a name='Prompt1Prompt2'></a>什么是 Prompt1 和 Prompt2？

Prompt1 和 Prompt2 都是系统提示（system prompt），区别在于插入点不同。因为测试发现当前模型对第一个 [Stts] token 非常敏感，所以需要两个提示。

- Prompt1 插入到第一个 [Stts] 之前
- Prompt2 插入到第一个 [Stts] 之后

### 6.2. <a name='Prefix'></a>什么是 Prefix？

Prefix 主要用于控制模型的生成能力，类似于官方示例中的 refine prompt。这个 prefix 中应该只包含特殊的非语素 token，如 `[laugh_0]`、`[oral_0]`、`[speed_0]`、`[break_0]` 等。

### 6.3. <a name='Style_p'></a>Style 中 `_p` 的区别是什么？

Style 中带有 `_p` 的使用了 prompt + prefix，而不带 `_p` 的则只使用 prefix。

### 6.4. <a name='--compile'></a>为什么开启了 `--compile` 很慢？

由于还未实现推理 padding 所以如果每次推理 shape 改变都可能触发 torch 进行 compile

> 暂时不建议开启

### 6.5. <a name='colab2its'></a>为什么 colab 里面非常慢只有 2 it/s ？

请确保使用 gpu 而非 cpu。

- 点击菜单栏 【修改】
- 点击 【笔记本设置】
- 选择 【硬件加速器】 => T4 GPU

## 7. <a name='-1'></a>离线整合包

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
