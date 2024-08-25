import copy
import json
from typing import Dict, List

from modules.core.pipeline.generate.dcls import SynthAudio, TTSBucket
from modules.core.spk.TTSSpeaker import TTSSpeaker


class Bucketizer:
    def __init__(self, segments: list[SynthAudio]) -> None:
        self.segments = segments

    def seg_hash(self, audio: SynthAudio):
        seg = audio.seg
        temp_seg = copy.deepcopy(seg)
        if isinstance(temp_seg.spk, TTSSpeaker):
            temp_seg.spk = str(temp_seg.spk.id)

        json_data = temp_seg.__dict__

        return hash(
            json.dumps(
                {
                    k: v
                    for k, v in json_data.items()
                    # NOTE: 这两个都不影响模型
                    if k != "text" and k != "duration_ms"
                },
                sort_keys=True,
            )
        )

    def build_buckets(self):
        buckets: Dict[str, List[SynthAudio]] = {"<break>": []}
        for segment in self.segments:
            if segment.seg._type == "break":
                buckets["<break>"].append(segment)
                continue
            key = self.seg_hash(segment)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(segment)

        break_segments = buckets.pop("<break>")
        audio_segments = list(buckets.values())
        # 根据 bucket 第一个 seg 的 index 排序，越小的越靠前
        audio_segments.sort(key=lambda x: self.segments.index(x[0]))

        return [
            TTSBucket(break_segments),
            *[TTSBucket(segments) for segments in audio_segments],
        ]
