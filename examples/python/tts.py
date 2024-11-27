from client import SAFClient

client = SAFClient()

audio = client.tts({"text": "你好，欢迎使用 Speech AI Forge 项目。"})

with open("example.mp3", "wb") as f:
    f.write(audio)
