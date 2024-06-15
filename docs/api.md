# API

使用 `launch.py` 脚本启动 `api` 服务之后，你可以在 `http://localhost:7870/docs` 下查看和简单测试 `api`

所有参数：

| 参数              | 类型   | 默认值      | 描述                                            |
| ----------------- | ------ | ----------- | ----------------------------------------------- |
| `--host`          | `str`  | `"0.0.0.0"` | 服务器主机地址                                  |
| `--port`          | `int`  | `8000`      | 服务器端口                                      |
| `--reload`        | `bool` | `False`     | 启用自动重载功能（用于开发）                    |
| `--compile`       | `bool` | `False`     | 启用模型编译                                    |
| `--lru_size`      | `int`  | `64`        | 设置请求缓存池的大小；设置为 0 禁用 `lru_cache` |
| `--cors_origin`   | `str`  | `"*"`       | 允许的 CORS 源，使用 `*` 允许所有源             |
| `--no_playground` | `bool` | `False`     | 关闭 playground 入口                            |
| `--no_docs`       | `bool` | `False`     | 关闭 docs 入口                                  |
| `--no_half`       | `bool` | `False`     | 使用 f32 全精度推理                             |
| `--off_tqdm`      | `bool` | `False`     | 关闭 tqdm 进度条                                |
| `--exclude`       | `str`  | `""`        | 排除不需要的 api                                |
| `--device_id`     | `str`  | `None`      | 指定使用 gpu device_id                          |
| `--use_cpu`       | `str`  | `None`      | 当前可选值 `"all"`                              |

launch.py 脚本启动成功后，你可以在 `/docs` 下检查 api 是否开启。

下面特殊接口的介绍

## <a name='OpenAIAPI:v1audiospeech'></a>OpenAI API: `v1/audio/speech`

openai 接口比较简单，`input` 为必填项，其余均可为空。

一个简单的请求示例如下：

```bash
curl http://localhost:7870/v1/audio/speech \
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
curl "http://localhost:7870/v1/text:synthesize" -X POST \
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
    "audioEncoding": "mp3"
  },
  "enhancerConfig": {
    "enabled": true
  }
}' -o response.json
```

## playground

启动 api 服务之后，在 `/playground` 下有一个非 gradio 的调试页面用于接口测试

![playground](./playground.png)

> 如果不需要此页面，启动服务的时候加上 `--no_playground` 即可
