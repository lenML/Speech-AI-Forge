# 🗣️ ChatTTS-Forge

ChatTTS-Forge 是一个功能强大的文本转语音生成工具，支持通过类 SSML 语法生成丰富的音频长文本，并提供全面的 API 服务，适用于各种场景。无论是个性化的语音生成，还是高效的说话人和风格管理，ChatTTS-Forge 都能满足你的需求。

你可以通过以下几种方式体验和部署 ChatTTS-Forge：

| 名称                     | 描述                     | 链接                                                                                                                                                             |
| ------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Huggingface 在线体验** | ChatTTS-Forge 在线体验   | [ChatTTS-Forge 在线体验](https://huggingface.co/spaces/lenML/ChatTTS-Forge)                                                                                      |
| **Colab 一键启动**       | 点击按钮，一键启动 Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenML/ChatTTS-Forge/blob/main/colab.ipynb) |
| **Docker 部署**          | 查看 docker 部分         | [Docker](#docker)                                                                                                                                                |
| **本地部署**             | 查看环境准备部分         | [本地部署](#本地部署)                                                                                                                                            |

## Features

- **风格提示词注入**: 灵活调整输出风格，通过注入提示词实现个性化。
- **全面的 API 服务**: 所有功能均通过 API 访问，集成方便。
- **友好的调试 GUI**: 独立于 Gradio 的 playground，简化调试流程。
- **OpenAI 风格 API**: `/v1/openai/audio/speech` 提供类似 OpenAI 的语音生成接口。
- **Google 风格 API**: `/v1/google/text:synthesize` 提供类似 Google 的文本合成接口。
- **类 SSML 支持**: 使用类 SSML 语法创建丰富的音频长文本。
- **说话人管理**: 通过名称或 ID 高效复用说话人。
- **风格管理**: 通过名称或 ID 复用说话风格，内置 32 种不同风格。
- **文本标准化**: 针对 ChatTTS 优化的文本标准化，解决大部分不支持的 token。
- **独立 refine API**: 提供单独的 refine 调试接口，提升调试效率。

## Interface

<table>
  <tr>
    <th>项目</th>
    <th>描述</th>
    <th>部署或使用方式</th>
    <th>图片</th>
  </tr>
  <tr>
    <td rowspan="2">API</td>
    <td>部署后打开 <code>http://localhost:8000/docs</code> 可查看详细信息。</td>
    <td>运行 <code>python launch.py</code></td>
    <td rowspan="2"><img src="./docs/api.png" alt="api"><br><img src="./docs/playground.png" alt="playground"></td>
  </tr>
  <tr>
    <td>实现了一套用于调试 API 的 Playground 前端页面，独立于 Python 代码非 Gradio。</td>
    <td>部署后打开 <code>http://localhost:8000/playground/index.html</code></td>
  </tr>
  <tr>
    <td>WebUI</td>
    <td>某些情况可能需要 WebUI（比如 HuggingFace/Colab），这里是一个简单实现。（WebUI中将不会支持对任何本地文件写操作。）</td>
    <td>运行 <code>python webui.py</code></td>
    <td><img src="./docs/webui.png" alt="webui"></td>
  </tr>
</table>

## 本地部署

> 预估显存消耗 3.7gb 左右

1. 克隆项目: `git clone https://github.com/lenML/ChatTTS-Forge.git`
2. 准备模型，放到如下目录

   ![model_dir](./docs/model_dir.png)

   - 自行下载（任选其一）

     - [HuggingFace](https://huggingface.co/2Noise/ChatTTS)
     - [ModelScope](https://modelscope.cn/models/pzc163/chatTTS/)

   - 使用脚本下载（任选其一）
     - HuggingFace: 执行 `python ./download_models.py --source huggingface`
     - ModelScope: 执行 `python ./download_models.py --source modelscope`

3. 安装 ffmpeg: `apt-get install ffmpeg`
4. 安装 rubberband: `apt-get install rubberband-cli`
5. 安装 Python 依赖: `python -m pip install -r requirements.txt`
6. 根据你的需求启动需要的服务，具体启动参数如下。

### launch.py

Launch.py 是 ChatTTS-Forge 的启动脚本，用于配置和启动 API 服务器。

所有参数：

| 参数            | 类型   | 默认值      | 描述                                            |
| --------------- | ------ | ----------- | ----------------------------------------------- |
| `--host`        | `str`  | `"0.0.0.0"` | 服务器主机地址                                  |
| `--port`        | `int`  | `8000`      | 服务器端口                                      |
| `--reload`      | `bool` | `False`     | 启用自动重载功能（用于开发）                    |
| `--compile`     | `bool` | `False`     | 启用模型编译                                    |
| `--lru_size`    | `int`  | `64`        | 设置请求缓存池的大小；设置为 0 禁用 `lru_cache` |
| `--cors_origin` | `str`  | `"*"`       | 允许的 CORS 源，使用 `*` 允许所有源             |

### webui.py

WebUI.py 是一个用于配置和启动 Gradio Web UI 界面的脚本。

所有参数：

| 参数            | 类型   | 默认值      | 描述                                               |
| --------------- | ------ | ----------- | -------------------------------------------------- |
| `--server_name` | `str`  | `"0.0.0.0"` | 服务器主机地址                                     |
| `--server_port` | `int`  | `7860`      | 服务器端口                                         |
| `--share`       | `bool` | `False`     | 启用共享模式，允许外部访问                         |
| `--debug`       | `bool` | `False`     | 启用调试模式                                       |
| `--auth`        | `str`  | `None`      | 用于认证的用户名和密码，格式为 `username:password` |

## SSML

[SSML readme](./docs/SSML.md)

## Speaking style

[style readme](./docs/sytles.md)

## FAQ

noop

# Docker

WIP 开发中

# References

- ChatTTS: https://github.com/2noise/ChatTTS
- PaddleSpeech: https://github.com/PaddlePaddle/PaddleSpeech
