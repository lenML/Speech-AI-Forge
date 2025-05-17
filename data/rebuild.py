"""
这个脚本用于重建说话人文件

流程：就是读取之后重写写入源文件

场景 tts spk 文件增加了一些属性，用这个脚本重建

*注意，修改结构可能导致某些数据丢失，最好备份再使用这个脚本
"""

import argparse
import json
import os
import sys

from modules.core.spk import TTSSpeaker, spk_mgr

for spk in spk_mgr.list_speakers():
    filename = spk_mgr.get_item_path(lambda spk0: spk0 == spk)
    filepath = os.path.join(os.path.dirname(__file__), "speakers", filename)
    print(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        json_str = spk.to_json_str()
        f.write(json_str)
