import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from modules.utils.constants import ROOT_DIR

BIN_DIR_PATH = Path("./ffmpeg")

if not os.path.exists(BIN_DIR_PATH):
    os.makedirs(BIN_DIR_PATH)

ffmpeg_path = os.path.join(ROOT_DIR, "ffmpeg")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

ffmpeg_installed = shutil.which("ffmpeg") is not None
rubberband_installed = shutil.which("rubberband") is not None


class Downloader:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def download_and_extract(self, url, extract_path):
        zip_path = os.path.join(self.temp_dir.name, url.split("/")[-1])
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        os.unlink(zip_path)

    def install_ffmpeg_on_windows(self):
        print("windows系统，安装ffmpeg...")

        if not ffmpeg_installed:
            ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z"
            self.download_and_extract(ffmpeg_url, self.temp_dir.name)

        if not rubberband_installed:
            rubberband_url = "https://breakfastquay.com/files/releases/rubberband-3.3.0-gpl-executable-windows.zip"
            self.download_and_extract(rubberband_url, self.temp_dir.name)

        need_files = [
            "ffmpeg.exe",
            "ffplay.exe",
            "ffprobe.exe",
            "rubberband-r3.exe",
            "rubberband.exe",
            "sndfile.dll",
        ]

        # 遍历查找，移动到 BIN_DIR_PATH
        for root, dirs, files in os.walk(self.temp_dir.name):
            for file in files:
                if file in need_files:
                    shutil.move(os.path.join(root, file), BIN_DIR_PATH)

        print("安装完成.")

    def install_ffmpeg_on_mac(self):
        if shutil.which("brew") is None:
            print("请先安装brew.")
            return
        if not ffmpeg_installed:
            print("安装ffmpeg...")
            os.system("brew install ffmpeg -y")
        else:
            print("ffmpeg已安装.")
        if not rubberband_installed:
            print("安装rubberband...")
            os.system("brew install rubberband -y")
        else:
            print("rubberband已安装.")
        print("安装完成.")

    def install_ffmpeg_on_linux(self):
        if shutil.which("apt-get") is None:
            print("请先安装apt-get.")
            return
        if not ffmpeg_installed:
            print("安装ffmpeg...")
            os.system("apt-get install ffmpeg libavcodec-extra -y")
        if not rubberband_installed:
            print("安装rubberband...")
            os.system("apt-get install rubberband-cli -y")
        print("安装完成.")

    def __del__(self):
        self.temp_dir.cleanup()


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == "__main__":
    downloader = Downloader()

    if os.name == "posix":
        if os.uname().sysname == "Darwin":
            downloader.install_ffmpeg_on_mac()
        elif os.uname().sysname == "Linux":
            downloader.install_ffmpeg_on_linux()
    elif os.name == "nt":
        downloader.install_ffmpeg_on_windows()
