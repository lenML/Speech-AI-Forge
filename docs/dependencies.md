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

音频后处理如加速减速提高音量等依赖以下库：

- **ffmpeg**
- **rubberband**

### Windows 安装

下载地址：

- ffmpeg: [下载](https://ffmpeg.org/download.html)
- rubberband: [下载](https://breakfastquay.com/rubberband/)

下载后将其放置在易于访问的目录中，并配置环境变量 PATH，以确保命令行可以识别相应的可执行文件。

### Linux 安装

在 Ubuntu 系统上，可以通过以下命令安装 `ffmpeg` 和 `rubberband-cli`：

```bash
sudo apt-get install ffmpeg
sudo apt-get install rubberband-cli
```

其他系统的安装方式类似。

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
