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

## playground

启动 api 服务之后，在 `/playground` 下有一个非 gradio 的调试页面用于接口测试

![playground](./playground.png)

> 如果不需要此页面，启动服务的时候加上 `--no_playground` 即可
