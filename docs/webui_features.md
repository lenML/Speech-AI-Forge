# ChatTTS-Forge WebUI Features

## webui.py

WebUI.py 是一个用于配置和启动 Gradio Web UI 界面的脚本。

所有参数：

| 参数                   | 类型   | 默认值      | 描述                                               |
| ---------------------- | ------ | ----------- | -------------------------------------------------- |
| `--server_name`        | `str`  | `"0.0.0.0"` | 服务器主机地址                                     |
| `--server_port`        | `int`  | `7860`      | 服务器端口                                         |
| `--share`              | `bool` | `False`     | 启用共享模式，允许外部访问                         |
| `--debug`              | `bool` | `False`     | 启用调试模式                                       |
| `--compile`            | `bool` | `False`     | 启用模型编译                                       |
| `--auth`               | `str`  | `None`      | 用于认证的用户名和密码，格式为 `username:password` |
| `--no_half`            | `bool` | `False`     | 使用 f32 全精度推理                                |
| `--off_tqdm`           | `bool` | `False`     | 关闭 tqdm 进度条                                   |
| `--tts_max_len`        | `int`  | `1000`      | TTS（文本到语音）的最大文本长度                    |
| `--ssml_max_len`       | `int`  | `2000`      | SSML（语音合成标记语言）的最大文本长度             |
| `--max_batch_size`     | `int`  | `8`         | TTS 的最大批处理大小                               |
| `--device_id`          | `str`  | `None`      | 指定使用 gpu device_id                             |
| `--use_cpu`            | `str`  | `None`      | 当前可选值 `"all"`                                 |
| `--webui_experimental` | `bool` | `False`     | 是否开启实验功能（不完善的功能）                   |
| `--language`           | `str`  | `zh-CN`     | 设置 webui 本地化                                  |
| `--api`                | `bool` | `False`     | 是否开启 API                                       |

> 从 webui.py 入口启动， 可与 api 同时启动，api 的配置在下方 launch.py 脚本参数中说明， 开启后可在 `http://localhost:7860/docs` 查看 api

> 由于 `MKL FFT doesn't support tensors of type: Half` 所以 `--use_cpu="all"` 时需要开启 `--no_half`

## TTS

该页面提供了一个强大的对话式文本转语音（TTS）模型接口，支持中英文混读和多说话人能力。用户可以通过调节各种参数生成高质量的语音输出。

### 使用流程

1. **采样参数设置**

   - **温度（Temperature）**: 使用滑块调整，范围 0.01 到 2.0，默认值为 0.3。
   - **Top P**: 使用滑块调整，范围 0.1 到 1.0，默认值为 0.7。
   - **Top K**: 使用滑块调整，范围 1 到 50，默认值为 20。
   - **批处理大小（Batch Size）**: 使用滑块调整，范围 1 到最大批处理大小，默认值为 4。

2. **风格选择**

   - 选择样式（Style）: 从下拉菜单中选择预设样式，默认值为`*auto`。

3. **说话人选择**

   - **选择说话人**: 可以从下拉菜单中选择预设说话人，或手动输入说话人名称或种子。
   - **上传说话人文件**: 通过上传文件来自定义说话人。

4. **推理种子**

   - 设置推理种子（Inference Seed），可以手动输入或点击按钮随机生成。

5. **Prompt 工程**

   - 输入 Prompt 1、Prompt 2 和前缀（Prefix）。
   - 上传音频提示（如果启用了实验性功能）。

6. **文本输入**

   - 输入需要转换的文本，注意字数限制和英文文本的特殊标记。

7. **示例选择**

   - 从下拉菜单中选择示例文本，快速填充输入框。

8. **生成音频**

   - 点击“生成音频”（Generate Audio）按钮，生成语音输出。
   - 可选：启用增强（Enhance）和去噪（De-noise）功能。

9. **文本优化**
   - 输入优化提示（Refine Prompt），点击优化文本（Refine Text）按钮，对文本进行优化。

![screenshot](./webui.png)

## SSML

🚧 施工中

## Spliter

🚧 施工中

## Speaker

### Speaker Creator

使用本面板快捷抽卡生成 speaker.pt 文件。

1. **生成说话人**：输入种子、名字、性别和描述。点击 "Generate speaker.pt" 按钮，生成的说话人配置会保存为.pt 文件。
2. **测试说话人声音**：输入测试文本。点击 "Test Voice" 按钮，生成的音频会在 "Output Audio" 中播放。
3. **随机生成说话人**：点击 "Random Speaker" 按钮，随机生成一个种子和名字，可以进一步编辑其他信息并测试。

![screenshot](./webui/spk_creator.png)

### Speaker Merger

在本面板中，您可以选择多个说话人并指定他们的权重，合成新的语音并进行测试。以下是各个功能的详细说明：

1. 选择说话人: 您可以从下拉菜单中选择最多四个说话人（A、B、C、D），每个说话人都有一个对应的权重滑块，范围从 0 到 10。权重决定了每个说话人在合成语音中的影响程度。
2. 合成语音: 在选择好说话人和设置好权重后，您可以在“Test Text”框中输入要测试的文本，然后点击“测试语音”按钮来生成并播放合成的语音。
3. 保存说话人: 您还可以在右侧的“说话人信息”部分填写新的说话人的名称、性别和描述，并点击“Save Speaker”按钮来保存合成的说话人。保存后的说话人文件将显示在“Merged Speaker”栏中，供下载使用。

![screenshot](./webui/spk_merger.png)

## FQA

1. 如何增加文本上限？
   配置 `.env.webui` 环境变量文件即可
