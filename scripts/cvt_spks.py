"""
这个脚本用来将老版本的 .pt 说话人文件转为新版本
"""

import os

from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker

for fp, spk in spk_mgr.speakers.items():
    print(fp)
    name = spk.name
    gender = spk.gender
    describe = spk.describe
    emb = spk.emb
    print(emb.shape)
    print(name)
    print(gender)
    print(describe)

    new_fp = fp.replace(".pt", ".spkv1.json")
    if os.path.exists(spk_mgr.filepath(new_fp)):
        print(f"{new_fp} exists, skip")
        continue
    new_spk = TTSSpeaker.empty()
    new_spk.set_name(name)
    new_spk.set_desc(describe)
    new_spk.set_gender(gender)
    new_spk.set_author("forge-builtin")
    new_spk.set_token(tokens=[emb], model_id="chat-tts")
    spk_mgr.create_item(item=new_spk, filename=new_fp)


spk_mgr.refresh()

for fp, spk in spk_mgr.items.items():
    print(fp)
    spk: TTSSpeaker = spk
    print(spk._data.meta)
    print(spk._data.meta.name)
    print(spk._data.meta.desc)
