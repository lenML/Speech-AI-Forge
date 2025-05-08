[cn](./README.md) | [en](./README.en.md) | [Discord Server](https://discord.gg/9XnXUhAy3t)

# 🍦 Speech-AI-Forge

Speech-AI-Forge 是一个围绕 TTS 生成模型开发的项目，实现了 API Server 和 基于 Gradio 的 WebUI。

![banner](./docs/banner.png)

你可以通过以下几种方式体验和部署 Speech-AI-Forge：

| -            | 描述                     | 链接                                                                                                                                                               |
| ------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **在线体验** | 部署于 HuggingFace 中    | [HuggingFace Spaces](https://huggingface.co/spaces/lenML/ChatTTS-Forge)                                                                                            |
| **一键启动** | 点击按钮，一键启动 Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenML/Speech-AI-Forge/blob/main/colab.ipynb) |
| **容器部署** | 查看 docker 部分         | [Docker](#docker)                                                                                                                                                  |
| **本地部署** | 查看环境准备部分         | [本地部署](#InstallationandRunning)                                                                                                                                |

## Installation and Running

首先，确保 [相关依赖](./docs/dependencies.md) 已经正确安装，并查看 [模型下载](#模型下载) 下载所需模型

启动：

```
python webui.py
```

### webui features

[点我看详细图文介绍](./docs/webui_features.md)

- **TTS (文本转语音)**: 提供多种强大的 TTS 功能
  - **音色切换 (Speaker Switch)**: 可选择不同音色
    - **内置音色**: 提供多个内置音色，包括 `27 ChatTTS` / `7 CosyVoice` 音色 + `1 参考音色`
    - **自定义音色上传**: 支持上传自定义音色文件并进行实时推理
    - **参考音色**: 支持上传参考音频/文本，直接基于参考音频进行 TTS 推理
  - **风格控制 (Style)**: 内置多种风格控制选项，调整语音风格
  - **长文本推理 (Long Text)**: 支持超长文本的推理，自动分割文本
    - **Batch Size**: 支持设置 `Batch size`，提升支持批量推理模型的长文本推理速度
  - **Refiner**: 支持 `ChatTTS` 原生文本 `refiner`，支持无限长文本处理
  - **分割器设置 (Splitter)**: 调整分割器配置，控制分割结束符（`eos`）和分割阈值
  - **调节器 (Adjuster)**: 支持调整 `速度/音调/音量`，并增加 `响度均衡` 功能，优化音频输出
  - **人声增强 (Voice Enhancer)**: 使用 `Enhancer` 模型增强 TTS 输出，提高语音质量
  - **生成历史 (Generation History)**: 保存最近三次生成结果，便于对比和选择
  - **多模型支持 (Multi-model Support)**: 支持多种 TTS 模型推理，包括 `ChatTTS` / `CosyVoice` / `FishSpeech` / `GPT-SoVITS` / `F5-TTS` 等

- **SSML (语音合成标记语言)**: 提供高级 TTS 合成控制工具
  - **分割器 (Splitter)**: 精细控制长文本的分割结果
  - **Podcast**: 帮助创建 `长文本`、`多角色` 的音频，适合博客或剧本式的语音合成
  - **From Subtitle**: 从字幕文件生成 SSML 脚本，方便一键生成语音
  - **脚本编辑器 (Script Editor)**: 新增 SSML 脚本编辑器，支持从分割器（Podcast、来自字幕）导出并编辑 SSML 脚本，进一步优化语音生成效果

- **音色管理 (Voice Management)**:
  - **音色构建器 (Builder)**: 创建自定义音色，可从 ChatTTS seed 创建音色，或使用参考音频生成音色
  - **试音功能 (Test Voice)**: 上传音色文件，进行简单的试音和效果评估
  - **ChatTTS 调试工具**: 专门针对 `ChatTTS` 音色的调试工具
    - **音色抽卡 (Random Seed)**: 使用随机种子抽取不同的音色，生成独特的语音效果
    - **音色融合 (Blend)**: 融合不同种子创建的音色，获得新的语音效果
  - **音色 Hub**: 从音色库中选择并下载音色到本地，访问音色仓库 [Speech-AI-Forge-spks](https://github.com/lenML/Speech-AI-Forge-spks) 获取更多音色资源

- **ASR (自动语音识别)**:
  - **Whisper**: 使用 Whisper 模型进行高质量的语音转文本（ASR）
  - **SenseVoice**: 正在开发中的 ASR 模型，敬请期待

- **工具 (Tools)**:
  - **后处理工具 (Post Process)**: 提供音频剪辑、调整和增强等功能，优化生成的语音质量


### `launch.py`: API Server

某些情况，你并不需要 webui 或者需要更高的 api 吞吐，那么可以使用这个脚本启动单纯的 api 服务。

启动：

```
python launch.py
```

启动之后开启 `http://localhost:7870/docs` 可以查看开启了哪些 api 端点

更多帮助信息:

- 通过 `python launch.py -h` 查看脚本参数
- 查看 [API 文档](./docs/api.md)

## Docker

### 镜像

WIP 开发中

### 手动 build

下载模型: `python -m scripts.download_models --source modelscope`

> 此脚本将下载 `chat-tts` 和 `enhancer` 模型，如需下载其他模型，请看后续的 `模型下载` 介绍

- webui: `docker-compose -f ./docker-compose.webui.yml up -d`
- api: `docker-compose -f ./docker-compose.api.yml up -d`

环境变量配置

- webui: [.env.webui](./.env.webui)
- api: [.env.api](./.env.api)


## 模型支持

| 模型类别        | 模型名称                                                                                       | 流式级别 | 支持多语言              | 实现情况           |
| --------------- | ---------------------------------------------------------------------------------------------- | -------- | ----------------------- | ------------------ |
| **TTS**         | [ChatTTS](https://github.com/2noise/ChatTTS)                                                  | token 级 | en, zh                  | ✅                 |
|                 | [FishSpeech](https://github.com/fishaudio/fish-speech)                                         | 句子级   | en, zh, jp, ko      | ✅ (1.4) |
|                 | [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)                                          | 句子级   | en, zh, jp, yue, ko     | ✅(v2)                 |
|                 | [FireRedTTS](https://github.com/FireRedTeam/FireRedTTS)                                        | 句子级   | en, zh                  | ✅                 |
|                 | [F5-TTS](https://github.com/SWivid/F5-TTS)                                                    | 句子级   | en, zh                  | ✅(v0.6/v1)                 |
|                 | [Index-TTS](https://github.com/index-tts/index-tts)                                           | 句子级   | en, zh                  | ✅                 |
|                 | GPTSoVits                                                                                      | 句子级   |                         | 🚧                 |
| **ASR**         | [Whisper](https://github.com/openai/whisper)                                                  | 🚧       | ✅                      | ✅                 |
|                 | [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)                                        | 🚧       | ✅                      | 🚧                 |
| **Voice Clone** | [OpenVoice](https://github.com/myshell-ai/OpenVoice)                                          |          |                         | ✅                 |
|                 | [RVC](https://github.com/svc-develop-team/RVC)                                                |          |                         | 🚧                 |
| **Enhancer**    | [ResembleEnhance](https://github.com/resemble-ai/resemble-enhance)                            |          |                         | ✅                 |


## 模型下载

由于 Forge 主要面向 API 功能开发，目前尚未实现自动下载逻辑，下载模型需手动调用下载脚本，具体脚本位于 `./scripts` 目录下。

### 下载脚本

| 功能         | 模型       | 下载命令                                                                  |
| ------------ | ---------- | ------------------------------------------------------------------------- |
| **TTS**      | ChatTTS    | `python -m scripts.dl_chattts --source huggingface`                       |
|              | FishSpeech(1.4) | `python -m scripts.downloader.fish_speech_1_4 --source huggingface`    |
|              | CosyVoice(v2)  | `python -m scripts.downloader.cosyvoice2 --source huggingface`            |
|              | FireRedTTS | `python -m scripts.downloader.fire_red_tts --source huggingface`          |
|              | Index-TTS | `python -m scripts.downloader.index_tts --source huggingface`          |
|              | F5-TTS(v0.6) | `python -m scripts.downloader.f5_tts --source huggingface`          |
|              | F5-TTS(v1) | `python -m scripts.downloader.f5_tts_v1 --source huggingface`          |
|              | F5-TTS(vocos) | `python -m scripts.downloader.vocos_mel_24khz --source huggingface`          |
| **ASR**      | Whisper    | `python -m scripts.downloader.faster_whisper --source huggingface`        |
| **CV**       | OpenVoice  | `python -m scripts.downloader.open_voice --source huggingface`            |
| **Enhancer** | 增强模型   | `python -m scripts.dl_enhance --source huggingface`                       |

> **注意**：如果需要使用 ModelScope 下载模型，请使用 `--source modelscope`。部分模型可能无法使用 ModelScope 下载。

## FAQ

### 如何语音复刻？

目前已经支持各个模型的语音复刻功能，且在 skpv1 格式中也适配了参考音频等格式，下面是几种方法使用语音复刻：

1. 在 webui 中：在音色选择栏可以上传参考音色，这里可以最简单的使用语音复刻功能
2. 使用 api 时：使用 api 需要通过音色（即说话人）来使用语音复刻功能，所以，首先你需要创建一个你需要的说话人文件（.spkv1.json），并在调用 api 时填入 spk 参数为说话人的 name，即可使用。
3. Voice Clone：现在还支持使用 voice clone 模型进行语音复刻，使用 api 时配置相应 `参考` 即可。（由于现目前只支持 OpenVoice 用于 voice clone，所以不需要指定模型名称）

相关讨论 #118

### 配置了参考音频的 spk 文件生成结果全是杂音？

很大可能是上传音频配置有问题，所以建议一下几个方式解决：

1. 更新：更新代码更新依赖库版本，最重要的是更新 gradio （不出意外的话推荐尽量用最新版本）
2. 处理音频：用 ffmpeg 或者其他软件编辑音频，转为单声道然后再上传，也可以尝试转码为 wav 格式
3. 检查文本：检查参考文本是否有不支持的字符。同时，建议参考文本使用 `"。"` 号结尾（这是模型特性 😂）
4. 用 colab 创建：可以考虑使用 `colab` 环境来创建 spk 文件，最大限度减少运行环境导致的问题
5. TTS 测试：目前 webui tts 页面里，你可以直接上传参考音频，可以先测试音频和文本，调整之后，再生成 spk 文件

### 可以训练模型吗？

现在没有，本库主要是提供推理服务框架。
有计划增加一些训练相关的功能，但是预计不会太积极的推进。

### 如何优化推理速度？

首先，无特殊情况本库只计划整合和开发工程化方案，而对于模型推理优化比较依赖上游仓库或者社区实现
如果有好的推理优化欢迎提 issue 和 pr

现目前，最实际的优化是开启多 workers，启动 `launch.py` 脚本时开启 `--workers N` 以增加服务吞吐

还有其他待选不完善的提速优化，有兴趣的可尝试探索：

1. compile: 模型都支持 compile 加速，大约有 30% 增益，但是编译期很慢
2. flash_attn：使用 flash attn 加速，有支持（`--flash_attn` 参数），但是也不完善
3. vllm：未实现，待上游仓库更新

### 什么是 Prompt1 和 Prompt2？

> 仅限 ChatTTS

Prompt1 和 Prompt2 都是系统提示（system prompt），区别在于插入点不同。因为测试发现当前模型对第一个 [Stts] token 非常敏感，所以需要两个提示。

- Prompt1 插入到第一个 [Stts] 之前
- Prompt2 插入到第一个 [Stts] 之后

### 什么是 Prefix？

> 仅限 ChatTTS

Prefix 主要用于控制模型的生成能力，类似于官方示例中的 refine prompt。这个 prefix 中应该只包含特殊的非语素 token，如 `[laugh_0]`、`[oral_0]`、`[speed_0]`、`[break_0]` 等。

### Style 中 `_p` 的区别是什么？

Style 中带有 `_p` 的使用了 prompt + prefix，而不带 `_p` 的则只使用 prefix。

### 为什么开启了 `--compile` 很慢？

由于还未实现推理 padding 所以如果每次推理 shape 改变都可能触发 torch 进行 compile

> 暂时不建议开启

### 为什么 colab 里面非常慢只有 2 it/s ？

请确保使用 gpu 而非 cpu。

- 点击菜单栏 【修改】
- 点击 【笔记本设置】
- 选择 【硬件加速器】 => T4 GPU

## 离线整合包

感谢 @Phrixus2023 提供的整合包：
https://pan.baidu.com/s/1Q1vQV5Gs0VhU5J76dZBK4Q?pwd=d7xu

相关讨论：
https://github.com/lenML/Speech-AI-Forge/discussions/65

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
- FireRedTTS: https://github.com/FireRedTeam/FireRedTTS
- F5-TTS: https://github.com/SWivid/F5-TTS

- Whisper: https://github.com/openai/whisper

- ChatTTS 默认说话人: https://github.com/2noise/ChatTTS/issues/238
