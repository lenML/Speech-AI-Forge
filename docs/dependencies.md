# dependencies

在启动服务之前，请确保已安装以下依赖项。

## 1. Python 和 Git

### Python 3.10 和 Git 安装方法：

- **Windows**:

  - 下载并运行 Python 3.10 的安装程序: [网页](https://www.python.org/downloads/release/python-3106/), [exe](https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe) 或 win7 版本。
  - 下载并运行 Git 的安装程序: [网页](https://git-scm.com/download/win)。

- **Linux (基于 Debian 的发行版)**:

  ```bash
  sudo apt install wget git python3 python3-venv
  ```

- **Linux (基于 Red Hat 的发行版)**:

  ```bash
  sudo dnf install wget git python3
  ```

- **Linux (基于 Arch 的发行版)**:
  ```bash
  sudo pacman -S wget git python3
  ```

## 2. 下载仓库代码

执行以下命令将本仓库代码克隆到本地：

```bash
git clone https://github.com/lenML/ChatTTS-Forge.git --depth=1
```

## 3. 模型文件

- 自行下载（任选其一）：

  - [HuggingFace](https://huggingface.co/2Noise/ChatTTS)
  - [ModelScope](https://modelscope.cn/models/pzc163/chatTTS/)

- 使用脚本下载（任选其一）：
  - 从 HuggingFace 下载：
    ```bash
    python -m scripts/download_models --source huggingface
    ```
  - 从 ModelScope 下载：
    ```bash
    python -m scripts/download_models --source modelscope
    ```

安装完成之后 `models` 文件夹应该如下

```
./models
├── ChatTTS
│   ├── asset
│   │   ├── DVAE.pt
│   │   ├── Decoder.pt
│   │   ├── GPT.pt
│   │   ├── Vocos.pt
│   │   ├── spk_stat.pt
│   │   └── tokenizer.pt
│   └── config
│       ├── decoder.yaml
│       ├── dvae.yaml
│       ├── gpt.yaml
│       ├── path.yaml
│       └── vocos.yaml
├── put_model_here
└── resemble-enhance
    ├── hparams.yaml
    └── mp_rank_00_model_states.pt
```

## 4. 后处理工具链

音频后处理操作（如加速、减速、提高音量等）依赖以下库：

- **ffmpeg** 或 **libav**（推荐使用 ffmpeg）
- **rubberband-cli**（仅 Linux 环境需要）

### 安装 ffmpeg

**Mac（使用 [Homebrew](http://brew.sh)）**:

```bash
brew install ffmpeg
```

**Linux（使用 aptitude）**:

```bash
apt-get install ffmpeg libavcodec-extra
apt-get install rubberband-cli
```

**Windows**:

1. 从[此处](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z)下载并解压 ffmpeg 的 Windows 二进制文件。
2. 将 ffmpeg 的`/bin`文件夹添加到您的环境变量`PATH`中。

## 5. 安装 python 依赖

### pytorch

由于 pytroch 安装与你的本机环境有关，请自行安装对应版本，下面是一个简单的安装脚本

> （如果直接运行某些情况可能会安装 cpu 版本，具体应该指定什么版本请自行确定）

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
```

### 其余依赖

```bash
python -m pip install -r requirements.txt
```

## 6. 其他

如果需要部署 Docker 环境，请配置 Docker 和 Docker Compose。

- Docker: https://docs.docker.com/get-docker/
- Docker Compose: https://docs.docker.com/compose/
