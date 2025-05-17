import tempfile

import gradio as gr
from moviepy.editor import VideoFileClip


def create_audio_separator():
    """
    创建音频分离组件
    """

    def extract_audio(video_path: str):
        # 使用 moviepy 加载视频文件
        video_clip = VideoFileClip(video_path)

        # 提取音频
        audio_clip = video_clip.audio

        # 在临时文件夹中保存音频
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3"
        ) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            audio_clip.write_audiofile(temp_audio_file_path)

        # 返回提取的音频路径和状态信息
        return temp_audio_file_path

    with gr.Row():
        with gr.Column(scale=3):
            input_video = gr.Video(label="Input Video", sources=["upload"])
        with gr.Column(scale=6):
            output_audio = gr.Audio(label="提取的音频")
            btn = gr.Button("提取音频", variant="primary")

        # 设置按钮点击后的回调函数
        btn.click(
            fn=extract_audio,
            inputs=[input_video],
            outputs=[output_audio],
        )
