# SAF-SSML

SAF-SSML 是类似 微软 tts 的那种格式，结合本系统中的 speaker 和 style 会很好用。

> 目前这个 SSML 系统是非常实验性质的实现，只具备基本功能
> SAF-SSML 规范暂时未定，未来可能随时修改

## What is SSML ?

SSML（Speech Synthesis Markup Language）是一种基于 XML 的标记语言，用于控制文本到语音合成（TTS）系统的转换过程。通过 SSML，用户可以指定语音的各种属性，如语速、音量、音调、停顿时间等，以生成更加自然和个性化的语音输出。

如果你想更深入了解 SSML，可以参考下面这两个文档，深入理解其如何在 TTS 系统中使用

- 微软 TTS 文档：https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/speech-synthesis-markup-structure
- 谷歌 TTS 文档：https://cloud.google.com/text-to-speech/docs/ssml?hl=zh-cn

## SAF-SSML v0.1

我们的 SSML 实现，目前只拥有以下三种元素

- voice: 指定说话人和对应参数
- prosody: 对指定生成片段进行 rate pitch volume 调整
- break: 插入空白

下面是一个简单的例子

```xml
<speak version="0.1">
    <voice spk="Bob" style="narration-relaxed">
        SAF 用于合成多角色多情感的有声书示例
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        黛玉冷笑道：
    </voice>
    <voice spk="female2" style="angry">
        我说呢 [uv_break] ，亏了绊住，不然，早就飞起来了。
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        宝玉道：
    </voice>
    <voice spk="Alice" style="unfriendly">
        “只许和你玩 [uv_break] ，替你解闷。不过偶然到他那里，就说这些闲话。”
    </voice>
    <voice spk="female2" style="angry">
        “好没意思的话！[uv_break] 去不去，关我什么事儿？ 又没叫你替我解闷儿 [uv_break]，还许你不理我呢”
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        说着，便赌气回房去了。
    </voice>
</speak>
```

[ssml_demo.webm](https://github.com/lenML/SAF-Forge/assets/37396659/b2434702-1e3c-4e2a-ae94-2012897e16d7)

## voice

| 属性      | 描述                                                              |
| --------- | ----------------------------------------------------------------- |
| spk       | 指定语音角色的 ID 或名称                                          |
| style     | 指定语音的风格，例如 "narration-relaxed"                          |
| seed      | 指定随机种子，用于生成不同的语音变体                              |
| top_p     | 控制生成文本的多样性，值越高多样性越强                            |
| top_k     | 限制生成时考虑的最高概率词数                                      |
| temp      | 控制生成的随机性，值越高随机性越强                                |
| prompt1   | 自定义提示 1，用于引导语音生成                                    |
| prompt2   | 自定义提示 2，用于引导语音生成                                    |
| prefix    | 在生成的文本前添加的前缀                                          |
| normalize | 是否对文本进行标准化处理，"True" 表示标准化，"False" 表示不标准化 |

```xml
<speak version="0.1">
    <voice spk="Bob" seed="42" temp="0.7">
        temperature for sampling ，这个值有可能被 style 或者 speaker 覆盖
    </voice>
</speak>
```

## prosody

| 属性   | 描述                                                     |
| ------ | -------------------------------------------------------- |
| rate   | 语速，1.0 表示正常语速，大于 1 表示加速，小于 1 表示减速 |
| volume | 音量，以分贝为单位，可以是正数或负数                     |
| pitch  | 音调，以半音为单位，可以是正数或负数                     |
| ...    | 并支持所有 voice 属性                                    |

> 受限于模型能力，暂时无法做到对单个字控制。尽量在一个 prosody 中用长文本效果更好。

prosody 和 voice 一样接收所有语音控制参数，除此之外还可以控制 rate volume pitch 以对生成语音进行细致的后处理。

一个例子如下

```xml
<speak version="0.1">
    <voice spk="Bob" style="narration-relaxed">
        使用 prosody 控制生成文本的语速语调和音量，示例如下

        <prosody>
            无任何限制将会继承父级voice配置进行生成
        </prosody>
        <prosody rate="1.5">
            设置 rate 大于1表示加速，小于1为减速
        </prosody>
        <prosody pitch="6">
            设置 pitch 调整音调，设置为6表示提高6个半音
        </prosody>
        <prosody volume="2">
            设置 volume 调整音量，设置为2表示提高2个分贝
        </prosody>

        在 voice 中无prosody包裹的文本即为默认生成状态下的语音
    </voice>
</speak>
```

[prosody_demo.webm](https://github.com/lenML/ChatTTS-Forge/assets/37396659/b5ad4c8d-f519-4b9a-bacf-290e4cc7d6df)

## break

| 属性 | 描述                                 |
| ---- | ------------------------------------ |
| time | 停顿的时间，单位为毫秒，例如 "500ms" |

空白，用于在文本中插入固定时长的空白停顿

```xml
<speak version="0.1">
    <voice spk="Bob" style="narration-relaxed">
        使用 break 标签将会简单的

        <break time="500" />

        插入一段空白到生成结果中
    </voice>
</speak>
```

[break_demo.webm](https://github.com/lenML/ChatTTS-Forge/assets/37396659/e1c682b8-fce1-40fa-a4bf-7465a266798a)
