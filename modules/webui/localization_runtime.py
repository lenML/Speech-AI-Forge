class LocalizationVars:
    def __init__(self):
        self.DEFAULT_TTS_TEXT = ""
        self.DEFAULT_SPEAKER_TEST_TEXT = ""
        self.DEFAULT_SPEAKER_MERAGE_TEXT = ""
        self.DEFAULT_SSML_TEXT = ""

        self.ssml_examples = []
        self.tts_examples = []
        self.podcast_default = []


class ZHLocalizationVars(LocalizationVars):
    def __init__(self):
        super().__init__()
        self.DEFAULT_TTS_TEXT = "chat T T S 是一款强大的对话式文本转语音模型。它有中英混读和多说话人的能力。"
        self.DEFAULT_SPEAKER_TEST_TEXT = (
            "说话人测试 123456789 [uv_break] ok, test done [lbreak]"
        )
        self.DEFAULT_SPEAKER_MERAGE_TEXT = (
            "说话人合并测试 123456789 [uv_break] ok, test done [lbreak]"
        )
        self.DEFAULT_SSML_TEXT = """
<speak version="0.1">
  <voice spk="Bob" seed="42" style="narration-relaxed">
    这里是一个简单的 SSML 示例 [lbreak] 
  </voice>
</speak>
        """.strip()

        self.ssml_examples = [
            """
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
""",
            """
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
""",
            """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        使用 break 标签将会简单的 [lbreak]
        
        <break time="500" />

        插入一段空白到生成结果中 [lbreak]
    </voice>
</speak>
""",
        ]

        self.tts_examples = [
            {
                "text": """
Fear is the path to the dark side. Fear leads to anger. Anger leads to hate. Hate leads to suffering.
恐惧是通向黑暗之路。恐惧导致愤怒。愤怒引发仇恨。仇恨造成痛苦。 [lbreak]
Do. Or do not. There is no try.
要么做，要么不做，没有试试看。[lbreak]
Peace is a lie, there is only passion.
安宁即是谎言，激情方为王道。[lbreak]
Through passion, I gain strength.
我以激情换取力量。[lbreak]
Through strength, I gain power.
以力量赚取权力。[lbreak]
Through power, I gain victory.
以权力赢取胜利。[lbreak]
Through victory, my chains are broken.
于胜利中超越自我。[lbreak]
The Force shall free me.
原力任我逍遥。[lbreak]
May the force be with you!
愿原力与你同在！[lbreak]
              """.strip()
            },
            {
                "text": "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
            },
            {
                "text": "Big 🍌, a big 🍌, hey, your feeling is really wonderful [lbreak]"
            },
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

        self.podcast_default = [
            [
                1,
                "female2",
                "你好，欢迎收听今天的播客内容。今天我们要聊的是中华料理。",
                "podcast",
            ],
            [
                2,
                "Alice",
                "嗨，我特别期待这个话题！中华料理真的是博大精深。",
                "podcast",
            ],
            [
                3,
                "Bob",
                "没错，中华料理有着几千年的历史，而且每个地区都有自己的特色菜。",
                "podcast",
            ],
            [
                4,
                "female2",
                "那我们先从最有名的川菜开始吧。川菜以其麻辣著称，是很多人的最爱。",
                "podcast",
            ],
            [
                5,
                "Alice",
                "对，我特别喜欢吃麻婆豆腐和辣子鸡。那种麻辣的感觉真是让人难以忘怀。",
                "podcast",
            ],
            [
                6,
                "Bob",
                "除了川菜，粤菜也是很受欢迎的。粤菜讲究鲜美，像是白切鸡和蒸鱼都是经典。",
                "podcast",
            ],
            [
                7,
                "female2",
                "对啊，粤菜的烹饪方式比较清淡，更注重食材本身的味道。",
                "podcast",
            ],
            [
                8,
                "Alice",
                "还有北京的京菜，像北京烤鸭，那可是来北京必吃的美食。",
                "podcast",
            ],
            [
                9,
                "Bob",
                "不仅如此，还有淮扬菜、湘菜、鲁菜等等，每个菜系都有其独特的风味。",
                "podcast",
            ],
            [
                10,
                "female2",
                "对对对，像淮扬菜的狮子头，湘菜的剁椒鱼头，都是让人垂涎三尺的美味。",
                "podcast",
            ],
        ]


class ENLocalizationVars(LocalizationVars):
    def __init__(self):
        super().__init__()
        self.DEFAULT_TTS_TEXT = "Chat T T S is a powerful conversational text-to-speech model. It has the ability to mix Chinese and English and multiple speakers."
        self.DEFAULT_SPEAKER_TEST_TEXT = (
            "Speaker test 123456789 [uv_break] ok, test done [lbreak]"
        )
        self.DEFAULT_SPEAKER_MERAGE_TEXT = (
            "Speaker merge test 123456789 [uv_break] ok, test done [lbreak]"
        )
        self.DEFAULT_SSML_TEXT = """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        Here is a simple SSML example [lbreak]
    </voice>
</speak>
        """.strip()

        self.ssml_examples = [
            """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        Below is an example of ChatTTS synthesizing an audiobook with multiple roles and emotions [lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        Daiyu sneered: [lbreak]
    </voice>
    <voice spk="female2" seed="42" style="angry">
        I said [uv_break], it's a loss to trip, otherwise, I would have flown up long ago [lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        Bao Yu said: [lbreak]
    </voice>
    <voice spk="Alice" seed="42" style="unfriendly">
        "Only play with you [uv_break], to relieve your boredom. But occasionally go to his place, just say these idle words." [lbreak]
    </voice>
    <voice spk="female2" seed="42" style="angry">
        "What a boring thing! [uv_break] Go or not, it's none of my business? I didn't ask you to relieve my boredom [uv_break], and you don't even care about me." [lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        Saying that, he went back to the room in anger [lbreak]
    </voice>
</speak>""",
        ]

        self.tts_examples = [
            {
                "text": "I guess it comes down a simple choice. Get busy living or get busy dying.",
            },
            {
                "text": "You got a dream, you gotta protect it. People can't do something themselves, they wanna tell you you can't do it. If you want something, go get it.",
            },
            {
                "text": "Don't ever let somebody tell you you can't do something. Not even me. Alright? You got a dream, you gotta protect it. When people can't do something themselves, they're gonna tell you that you can't do it. You want something, go get it. Period.",
            },
        ]
        self.podcast_default = [
            [
                1,
                "female2",
                "Hello, welcome to today's podcast. Today, we're going to talk about global cuisine.",
                "podcast",
            ],
            [
                2,
                "Alice",
                "Hi, I'm really excited about this topic! Global cuisine is incredibly diverse and fascinating.",
                "podcast",
            ],
            [
                3,
                "Bob",
                "Absolutely, every country has its own unique culinary traditions and specialties.",
                "podcast",
            ],
            [
                4,
                "female2",
                "Let's start with Italian cuisine. Italian food is loved worldwide, especially for its pasta and pizza.",
                "podcast",
            ],
            [
                5,
                "Alice",
                "Yes, I especially love a good Margherita pizza and a hearty plate of spaghetti carbonara. The flavors are simply amazing.",
                "podcast",
            ],
            [
                6,
                "Bob",
                "Besides Italian cuisine, Japanese cuisine is also very popular. Dishes like sushi and ramen have become global favorites.",
                "podcast",
            ],
            [
                7,
                "female2",
                "Exactly, Japanese cuisine is known for its emphasis on fresh ingredients and delicate presentation.",
                "podcast",
            ],
            [
                8,
                "Alice",
                "And then there's Mexican cuisine, with its bold flavors and colorful dishes like tacos and guacamole.",
                "podcast",
            ],
            [
                9,
                "Bob",
                "Not to mention, there's also Indian cuisine, Thai cuisine, French cuisine, and so many more, each with its own distinctive flavors and techniques.",
                "podcast",
            ],
            [
                10,
                "female2",
                "Yes, like Indian curry, Thai tom yum soup, and French croissants, these are all mouth-watering dishes that are loved by people all over the world.",
                "podcast",
            ],
        ]
