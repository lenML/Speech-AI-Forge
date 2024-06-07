example_texts = [
    {
        "text": "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
    },
    {"text": "Big 🍌, a big 🍌, hey, your feeling is really wonderful [lbreak]"},
    {
        "text": """
# 这是 markdown 标题

```
代码块将跳过
```

- **文本标准化**:
  - **Markdown**: 自动检测处理 markdown 格式文本。
  - **数字转写**: 自动将数字转为模型可识别的文本。
  - **Emoji 适配**: 自动翻译 emoji 为可读文本。
  - **基于分词器**: 基于 tokenizer 预处理文本，覆盖模型所有不支持字符范围。
  - **中英文识别**: 适配英文环境。
        """
    },
    {
        "text": "天气预报显示，今天会有小雨，请大家出门时记得带伞。降温的天气也提醒我们要适时添衣保暖 [lbreak]",
    },
    {
        "text": "公司的年度总结会议将在下周三举行，请各部门提前准备好相关材料，确保会议顺利进行 [lbreak]",
    },
    {
        "text": "今天的午餐菜单包括烤鸡、沙拉和蔬菜汤，大家可以根据自己的口味选择适合的菜品 [lbreak]",
    },
    {
        "text": "请注意，电梯将在下午两点进行例行维护，预计需要一个小时的时间，请大家在此期间使用楼梯 [lbreak]",
    },
    {
        "text": "图书馆新到了一批书籍，涵盖了文学、科学和历史等多个领域，欢迎大家前来借阅 [lbreak]",
    },
    {
        "text": "电影中梁朝伟扮演的陈永仁的编号27149 [lbreak]",
    },
    {
        "text": "这块黄金重达324.75克 [lbreak]",
    },
    {
        "text": "我们班的最高总分为583分 [lbreak]",
    },
    {
        "text": "12~23 [lbreak]",
    },
    {
        "text": "-1.5~2 [lbreak]",
    },
    {
        "text": "她出生于86年8月18日，她弟弟出生于1995年3月1日 [lbreak]",
    },
    {
        "text": "等会请在12:05请通知我 [lbreak]",
    },
    {
        "text": "今天的最低气温达到-10°C [lbreak]",
    },
    {
        "text": "现场有7/12的观众投出了赞成票 [lbreak]",
    },
    {
        "text": "明天有62％的概率降雨 [lbreak]",
    },
    {
        "text": "随便来几个价格12块5，34.5元，20.1万 [lbreak]",
    },
    {
        "text": "这是固话0421-33441122 [lbreak]",
    },
    {
        "text": "这是手机+86 18544139121 [lbreak]",
    },
]

ssml_example1 = """
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
"""
ssml_example2 = """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        使用 prosody 控制生成文本的语速语调和音量，示例如下 [lbreak]

        <prosody>
            无任何限制将会继承父级voice配置进行生成 [lbreak]
        </prosody>
        <prosody rate="1.5">
            设置 rate 大于1表示加速，小于1为减速 [lbreak]
        </prosody>
        <prosody pitch="6">
            设置 pitch 调整音调，设置为6表示提高6个半音 [lbreak]
        </prosody>
        <prosody volume="2">
            设置 volume 调整音量，设置为2表示提高2个分贝 [lbreak]
        </prosody>

        在 voice 中无prosody包裹的文本即为默认生成状态下的语音 [lbreak]
    </voice>
</speak>
"""
ssml_example3 = """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        使用 break 标签将会简单的 [lbreak]
        
        <break time="500" />

        插入一段空白到生成结果中 [lbreak]
    </voice>
</speak>
"""

ssml_example4 = """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="excited">
        temperature for sampling (may be overridden by style or speaker) [lbreak]
        <break time="500" />
        温度值用于采样，这个值有可能被 style 或者 speaker 覆盖  [lbreak]
        <break time="500" />
        temperature for sampling ，这个值有可能被 style 或者 speaker 覆盖  [lbreak]
        <break time="500" />
        温度值用于采样，(may be overridden by style or speaker) [lbreak]
    </voice>
</speak>
"""

ssml_examples = [
    ssml_example1,
    ssml_example2,
    ssml_example3,
    ssml_example4,
]

default_ssml = """
<speak version="0.1">
  <voice spk="Bob" seed="42" style="narration-relaxed">
    这里是一个简单的 SSML 示例 [lbreak] 
  </voice>
</speak>
"""
