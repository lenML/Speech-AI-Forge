import logging
import os

from modules.ffmpeg_env import setup_ffmpeg_path
from modules.repos_static.sys_paths import setup_repos_paths

try:
    setup_repos_paths()
    setup_ffmpeg_path()
    # NOTE: 因为 logger 都是在模块中初始化，所以这个 config 必须在最前面
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
except BaseException:
    pass

import argparse
import json
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.core.pipeline.processor import NP_AUDIO
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.utils import audio_utils

_config = {
    "overwrite": False,
    "enhance": False,
    "normalize": False,
    "remove_silence": False,
    "reprocess": False,
}


# 读取文件夹即其中的文件，解析为dict，为生成spk做准备
def read_spks_configs(spk_dir: Path = Path("./scripts/spk/spk_factory")) -> list[dict]:
    spk_configs = []

    folders = [f for f in spk_dir.iterdir() if f.is_dir()]
    # 跳过以 .或者_ 开头的文件夹
    folders = [f for f in folders if not f.name.startswith(".")]
    folders = [f for f in folders if not f.name.startswith("_")]

    for folder in tqdm(folders, desc="Processing speakers"):
        config = {
            "name": folder.name,
            "author": None,
            "avatar": None,
            "gender": None,
            "desc": None,
            "version": None,
            "tags": [],
            "audio": [],
        }

        config_path = folder / "config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                user_config = json.load(f)
                config.update({k: user_config.get(k, v) for k, v in config.items()})

        for file in folder.iterdir():
            if file.suffix.lower() not in [".wav", ".mp3"]:
                continue
            if ".enhanced" in str(file):
                # 增强输出的音频不需要处理
                continue

            emotion = file.stem
            txt_file = file.with_suffix(".txt")
            caption = (
                txt_file.read_text(encoding="utf-8").strip()
                if txt_file.exists()
                else ""
            )

            audio_path = str(file.resolve())

            need_postprocess = (
                _config["enhance"] or _config["normalize"] or _config["remove_silence"]
            )
            if need_postprocess:
                audio_path = postprocess_audio(audio_path)

            config["audio"].append(
                {
                    "filepath": audio_path,
                    "caption": caption,
                    "emotion": emotion,
                }
            )

        if config["audio"]:
            spk_configs.append(config)

    return spk_configs


def create_spk_from_config(config: dict):
    spk = TTSSpeaker.empty()  # create empty spk
    if config["name"]:
        spk.set_name(config["name"])
    if config["name"]:
        spk.set_author(config["author"])
    if config["avatar"]:
        spk.set_avatar(config["avatar"])
    if config["desc"]:
        spk.set_desc(config["desc"])
    if config["version"]:
        spk.set_version(config["version"])
    if config["tags"]:
        spk.set_tags(config["tags"])
    if config["gender"]:
        spk.set_gender(config["gender"])
    for audio_file in config["audio"]:
        filepath = audio_file["filepath"]
        caption = audio_file["caption"]
        emotion = audio_file["emotion"]

        audio_ref = TTSSpeaker.create_spk_ref_from_filepath(filepath, caption)
        audio_ref.emotion = emotion

        spk.add_ref(ref=audio_ref)

    return spk


def save_spk_to_data(spk: TTSSpeaker):
    """
    将spk保存到data目录下面
    """
    json_str = spk.to_json_str()
    filepath = Path("./data/speakers") / (f"{spk.name}.spkv1.json")
    if not _config["overwrite"] and filepath.exists():
        print(f"[skip] file exists: {filepath}")
        return
    filepath.write_text(json_str, encoding="utf-8")


# NP_AUDIO = Tuple[int, npt.NDArray]
def postprocess_audio(filepath: str) -> str:
    """
    调用模型增强音频，并将增强后的音频保存为 filename.mp3 => filename.enhanced.mp3
    """
    out_path = Path(filepath)
    new_path = out_path.with_name(out_path.stem + ".enhanced" + out_path.suffix)

    reprocess = _config["reprocess"]
    enhance = _config["enhance"]
    normalize = _config["normalize"]
    remove_silence = _config["remove_silence"]

    if reprocess is False and new_path.exists():
        print(f"[skip postprocess] file exists: {new_path}")
        return str(new_path)

    audio = audio_utils.load_audio(filepath)
    ctx = TTSPipelineContext(
        enhancer_config=EnhancerConfig(
            enabled=enhance,
            model="resemble-enhance",
            # 还有其他值可以改，但是暂时用默认的应该够了
            # nfe=32,
            # solver="midpoint",
            # lambd=0.5,
            # tau=0.5,
        ),
        adjust_config=AdjustConfig(normalize=normalize, remove_silence=remove_silence),
    )
    pipeline = PipelineFactory.create_postprocess_pipeline(audio=audio, ctx=ctx)
    sr, data = pipeline.generate()

    sf.write(new_path, data, sr)
    return str(new_path)


def main():
    configs = read_spks_configs()
    for cfg in tqdm(configs, desc="Creating speakers"):
        try:
            spk = create_spk_from_config(cfg)
            save_spk_to_data(spk)
        except Exception as e:
            name = cfg["name"]
            print(f"音色保存失败 {name}")
            print(e)


if __name__ == "__main__":
    """
    这个脚本用于简化 音色创建 的流程

    此目录用于结合一键创建音色脚本创建音色使用。

    用法：
    1. 创建文件夹 每个文件夹对应一个音色
    2. 将音色音频放入文件夹中，文件名即为emotion感情标注，支持多感情，并对应文件名，创建 x.txt 标注音频内容，标注也可为空不创建，但是并不是所有模型都支持空标注复刻
    3. 【可选】可以在音色文件夹下创建 config.json 文件，其中可以包含 spk 的基础信息，比如名字、作者、版本、备注之类的，这里面如果有名字，将会覆盖文件夹名作为音色名
    4. 执行脚本
    5. 最终将输出到项目目录下的 `data/speakers` 文件夹中

    *当然，你也可以在创建音色 json 之后手动修改 json 中的内容，而不是用 config.json
    * --force 将直接覆盖音色文件，默认为跳过不生成
    * --enchance 将调用人声增强模型优化音色文件，可能增加处理时间
    * --normalize 将调用响度均衡模型优化音色文件

    文件夹目录示例：
      ├── a
      │   ├── default.wav
      │   ├── default.txt
      │   ├── happy.mp3
      │   ├── happy.txt
      │   └── config.json
      ├── b
      │   ├── default.wav
      │   ├── default.txt
      │   ├── angry.txt
      │   └── angry.wav
      ├── readme.md
      └── spk_facotry.py

    # 音质不够好
    `python -m scripts.spk.spk_factory.spk_factory --overwrite --enhance --normalize --remove_silence`

    # 音质可以但是需要处理裁剪
    `python -m scripts.spk.spk_factory.spk_factory --overwrite --normalize --remove_silence`

    # 音质很好也不需要裁剪
    `python -m scripts.spk.spk_factory.spk_factory --overwrite`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有音色文件")
    parser.add_argument("--enhance", action="store_true", help="使用人声增强")
    parser.add_argument("--normalize", action="store_true", help="使用响度均衡")
    parser.add_argument("--remove_silence", action="store_true", help="使用响度均衡")
    parser.add_argument(
        "--reprocess", action="store_true", help="是否重新生成增强音频文件"
    )
    args = parser.parse_args()

    _config["enhance"] = args.enhance
    _config["normalize"] = args.normalize
    _config["remove_silence"] = args.remove_silence
    _config["reprocess"] = args.reprocess
    _config["overwrite"] = args.overwrite

    main()
