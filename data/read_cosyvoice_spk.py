import torch

from modules.core.spk import TTSSpeaker, spk_mgr
from modules.core.spk.dcls import DcSpkVoiceToken

spk2info: dict = torch.load(
    "./models/CosyVoice_300M_Instruct/spk2info.pt", map_location="cpu"
)

for k, v in spk2info.items():
    print(k)
    embedding = v["embedding"]
    speech_token = v["speech_token"]
    speech_feat = v["speech_feat"]

    spk = TTSSpeaker.empty()
    spk.set_name(k)
    if k.endswith("女"):
        spk.set_gender("female")
    else:
        spk.set_gender("male")

    token = DcSpkVoiceToken(
        model_id="cosyvoice_300m_instruct",
        tokens=[speech_token],
        embedding=[embedding],
        feat=[speech_feat],
    )
    spk.set_token_obj(token=token)

    spk_mgr.create_item(spk, f"cv_{k}.spkv1.json")
