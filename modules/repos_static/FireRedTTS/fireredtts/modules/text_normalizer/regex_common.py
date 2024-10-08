import re

kaomoji_regex = re.compile(
    r"[oヽwΣ┗╰O︿Ψ凸]?[(|≡*（].{0,4}[Д✿_▽→≧﹏`∩⊙∇☆≡๑〃′エ≦▔＠﹁εヘ•́ω益‿≖ฺ皿•̀艹￣△|ﾟ].{0,5}[|≡*)）][┛ブ凸cｄd︴oOΨ︿w╯ノ]?"
)
chinese_regex = re.compile(r"[\u4e00-\u9fa5]")
digit_regex = re.compile(r"(\\d+)(\\.\\d+)?", re.UNICODE)

chinese_char_regex = re.compile(r"^[\u4e00-\u9fa5]$", re.UNICODE)
eng_and_digit_char_regex = re.compile(r"^[0-9.,A-Za-z]+$", re.UNICODE)
upper_eng_and_digit_regex = re.compile(r"^[ 0-9A-Z\"'.,:?!\-]+$", re.UNICODE)
valid_char_regex = re.compile(
    r"[\t\r\n ]|"
    r"[\u4e00-\u9fa5]|"
    r"\u0080|[\u20a0-\u20bf]|\u00a2|\u00a3|\u00a5|\uffe0|\uffe1|\uffe5|\uffe6|"
    r"\u3000|\u3002|\u00b7|\u2014|\u2019|\u2026|\uff01|\uff1f|\uff0e|\uff1a|\uff1b|\uff0b|\uff0c|\uff0d|\uff0f|[\ufe10-\ufe16]|[\ufe50-\ufe51]|[\ufe55-\ufe57]|\ufe6a|"
    r"[\u0030-\u0039]|"
    r"[\u0391-\u03c9]|"
    r"[\u00b0-\u00b3]|[\u2015-\u2018]|[\u3000-\u303f]|"
    r"[\u0022-\u002f\u003a-\u003e\u0040\u005b-\u0060\u007b-\u007e]|"
    r"[\uff21-\uff3a]|[\uff41-\uff5a]|[\u0041-\u005a]|[\u0061-\u007a]",
    re.UNICODE,
)
