# ChatTTS-SSML

ChatTTS-SSML 是类似 微软 tts 的那种格式，结合本系统中的 speaker 和 style 会很好用。

> 目前这个 SSML 系统是非常实验性质的实现，只具备基本功能
> ChatTTS-SSML 规范暂时未定，未来可能随时修改

## What is SSML ?

如果你不清楚什么是 SSML，可以参考下面这两个文档，大概理解 SSML 如何在 TTS 系统中如何使用

- 微软 TTS 文档：https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/speech-synthesis-markup-structure
- 谷歌 TTS 文档：https://cloud.google.com/text-to-speech/docs/ssml?hl=zh-cn

## ChatTTS-SSML v0.1

我们的 SSML 实现，目前只拥有以下三种元素

- voice: 指定说话人和对应参数
- prosody: 对指定生成片段进行 rate pitch volume 调整
- break: 插入空白

下面是一个简单的例子

```xml
<speak version="0.1">
    <voice spk="Bob" style="narration-relaxed">
        ChatTTS 用于合成多角色多情感的有声书示例
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

[ssml_demo.webm](https://github.com/lenML/ChatTTS-Forge/assets/37396659/b2434702-1e3c-4e2a-ae94-2012897e16d7)

## prosody

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
