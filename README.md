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

首先，确保 [相关依赖](./docs/dependencies.md) 已经正确安装

启动：

```
python webui.py
```

### webui features

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
- 音色 (说话人)：
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

#### How to link to SillyTavern?

通过 `/v1/xtts_v2` 系列 api，你可以方便的将 ChatTTS-Forge 连接到你的 SillyTavern 中。

下面是一个简单的配置指南:

1. 点开 插件拓展
2. 点开 `TTS` 插件配置部分
3. 切换 `TTS Provider` 为 `XTTSv2`
4. 勾选 `Enabled`
5. 选择/配置 `Voice`
6. **[关键]** 设置 `Provider Endpoint` 到 `http://localhost:7870/v1/xtts_v2`

![sillytavern_tts](./docs/sillytavern_tts.png)

## demo

### 风格化控制

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
  
[多角色.webm](https://github.com/lenML/Speech-AI-Forge/assets/37396659/82d91409-ad71-42ac-a4cd-d9c9340e3a07)

</details>

### 长文本生成

<details>
<summary>input</summary>

中华美食，作为世界饮食文化的瑰宝，以其丰富的种类、独特的风味和精湛的烹饪技艺而闻名于世。中国地大物博，各地区的饮食习惯和烹饪方法各具特色，形成了独树一帜的美食体系。从北方的京鲁菜、东北菜，到南方的粤菜、闽菜，无不展现出中华美食的多样性。

在中华美食的世界里，五味调和，色香味俱全。无论是辣味浓郁的川菜，还是清淡鲜美的淮扬菜，都能够满足不同人的口味需求。除了味道上的独特，中华美食还注重色彩的搭配和形态的美感，让每一道菜品不仅是味觉的享受，更是一场视觉的盛宴。

中华美食不仅仅是食物，更是一种文化的传承。每一道菜背后都有着深厚的历史背景和文化故事。比如，北京的烤鸭，代表着皇家气派；而西安的羊肉泡馍，则体现了浓郁的地方风情。中华美食的精髓在于它追求的“天人合一”，讲究食材的自然性和烹饪过程中的和谐。

总之，中华美食博大精深，其丰富的口感和多样的烹饪技艺，构成了一个充满魅力和无限可能的美食世界。无论你来自哪里，都会被这独特的美食文化所吸引和感动。

</details>

<details open>
<summary>output</summary>

[long_text_demo.webm](https://github.com/lenML/Speech-AI-Forge/assets/37396659/fe18b0f1-a85f-4255-8e25-3c953480b881)

</details>

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

## Roadmap

### Model Supports

#### TTS

| 模型名称   | 流式级别 | 支持复刻 | 支持训练 | 支持 prompt | 实现情况               |
| ---------- | -------- | -------- | -------- | ----------- | ---------------------- |
| ChatTTS    | token 级 | ✅       | ❓       | ❓          | ✅                     |
| FishSpeech | 句子级   | ✅       | ❓       | ❓          | ✅ (SFT 版本开发中 🚧) |
| CosyVoice  | 句子级   | ✅       | ❓       | ✅          | ✅                     |
| FireRedTTS | 句子级   | ✅       | ❓       | ✅          | ✅                     |
| GPTSoVits  | 句子级   | ✅       | ❓       | ❓          | 🚧                     |

#### ASR

| 模型名称   | 流式识别 | 支持训练 | 支持多语言 | 实现情况 |
| ---------- | -------- | -------- | ---------- | -------- |
| Whisper    | ✅       | ❓       | ✅         | ✅       |
| SenseVoice | ✅       | ❓       | ✅         | 🚧       |

#### Voice Clone

| 模型名称  | 实现情况 |
| --------- | -------- |
| OpenVoice | ✅       |
| RVC       | 🚧       |

#### Enhancer

| 模型名称        | 实现情况 |
| --------------- | -------- |
| ResembleEnhance | ✅       |

## 模型下载

由于 Forge 主要面向 API 功能开发，目前尚未实现自动下载逻辑，下载模型需手动调用下载脚本，具体脚本位于 `./scripts` 目录下。

### 下载脚本

| 功能         | 模型       | 下载命令                                                                  |
| ------------ | ---------- | ------------------------------------------------------------------------- |
| **TTS**      | ChatTTS    | `python -m scripts.dl_chattts --source huggingface`                       |
|              | FishSpeech | `python -m scripts.downloader.fish_speech_1_2sft --source huggingface`    |
|              | CosyVoice  | `python -m scripts.downloader.dl_cosyvoice_instruct --source huggingface` |
|              | FireRedTTS | `python -m scripts.downloader.fire_red_tts --source huggingface`          |
| **ASR**      | Whisper    | `python -m scripts.downloader.faster_whisper --source huggingface`        |
| **CV**       | OpenVoice  | `python -m scripts.downloader.open_voice --source huggingface`            |
| **Enhancer** | 增强模型   | `python -m scripts.dl_enhance --source huggingface`                       |

> **注意**：如果需要使用 ModelScope 下载模型，请使用 `--source modelscope`。部分模型可能无法使用 ModelScope 下载。

> **关于 CosyVoice**：不太确定应该使用哪个模型。整体来看，`instruct` 模型功能最多，但可能质量不是最佳。如果需要使用其他模型，请自行选择 `dl_cosyvoice_base.py`、`dl_cosyvoice_instruct.py` 或 `sft` 脚本。加载优先级为 `base` > `instruct` > `sft`，可根据文件夹存在性判断加载顺序。

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
- Whisper: https://github.com/openai/whisper

- ChatTTS 默认说话人: https://github.com/2noise/ChatTTS/issues/238
