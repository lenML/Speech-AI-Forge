# API

使用 `launch.py` 脚本启动 `api` 服务之后，你可以在 `http://localhost:8000/docs` 下查看和简单测试 `api`

下面特殊接口的介绍

## <a name='OpenAIAPI:v1audiospeech'></a>OpenAI API: `v1/audio/speech`

openai 接口比较简单，`input` 为必填项，其余均可为空。

一个简单的请求示例如下：

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer anything_your_wanna" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chattts-4w",
    "input": "Today is a wonderful day to build something people love! [lbreak]",
    "voice": "female2",
    "style": "chat"
  }' \
  --output speech.mp3
```

也可以使用 openai 库调用，具体可以看 [openai 官方文档](https://platform.openai.com/docs/guides/text-to-speech)

## <a name='GoogleAPI:v1text:synthesize'></a>Google API: `/v1/text:synthesize`

google 接口略复杂，但是某些时候用这个是必要的，因为这个接口将会返回 base64 格式的 audio

一个简单的请求示例如下：

```bash
curl "http://localhost:8000/v1/text:synthesize" -X POST \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
  "input": {
    "text": "Hello, ChatTTS Forage Google Endpoint Test. [lbreak]"
  },
  "voice": {
    "languageCode": "zh-CN",
    "name": "female2",
    "temperature": 0.3,
    "topP": 0.7,
    "topK": 20,
    "seed": 42
  },
  "audioConfig": {
    "audioEncoding": "MP3"
  }
}' -o response.json
```

## <a name='webui.py:WebUI'></a>`webui.py`: WebUI

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
| `--half`               | `bool` | `False`     | 开启 f16 半精度推理                                |
| `--off_tqdm`           | `bool` | `False`     | 关闭 tqdm 进度条                                   |
| `--tts_max_len`        | `int`  | `1000`      | TTS（文本到语音）的最大文本长度                    |
| `--ssml_max_len`       | `int`  | `2000`      | SSML（语音合成标记语言）的最大文本长度             |
| `--max_batch_size`     | `int`  | `8`         | TTS 的最大批处理大小                               |
| `--device_id`          | `str`  | `None`      | 指定使用 gpu device_id                             |
| `--use_cpu`            | `str`  | `None`      | 当前可选值 `"all"`                                 |
| `--webui_experimental` | `bool` | `False`     | 是否开启实验功能（不完善的功能）                   |
| `--language`           | `str`  | `zh-CN`     | 设置 webui 本地化                                  |
