import subprocess
from functools import lru_cache


@lru_cache()
def ffmpeg_version():
    try:
        result = subprocess.check_output(
            ["ffmpeg", "-version"], shell=False, encoding="utf8"
        )
        version_info = result.split("\n")[0]
        version_info = version_info.split("ffmpeg version")[1].strip()
        version_info = version_info.split("Copyright")[0].strip()
        return version_info
    except Exception:
        return "<none>"


if __name__ == "__main__":
    print(ffmpeg_version())
