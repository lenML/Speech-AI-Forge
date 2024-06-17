import json

from modules.speaker import speaker_mgr

# 出处: https://github.com/2noise/ChatTTS/issues/238
data = json.load(open("./data/slct_voice_240605.json", "r"))

print("load speakers: ", len(data))

for id, spk in data.items():
    print(id, spk["describe"])
    name = f"spk_{id}"
    describe = spk["describe"]
    gender = spk["gender"]
    speaker_mgr.create_speaker_from_tensor(
        filename=name,
        name=describe,
        describe=describe,
        gender=gender,
        tensor=spk["tensor"],
    )
