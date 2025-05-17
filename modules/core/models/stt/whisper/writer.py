import dataclasses
import io
import json
from typing import Iterable, NamedTuple, Optional, TextIO

import tqdm

from modules.core.models.stt.whisper.SegmentNormalizer import (
    SegmentNormalizer,
    SubtitleSegment,
)
from modules.core.models.stt.whisper.whisper_dcls import SttResult


class ResultWriter:
    always_include_hours: bool = True
    decimal_marker: str = "."

    def __init__(self, output: TextIO = None):
        self.output: TextIO = output or io.StringIO()
        self.subtitles: list[SubtitleSegment] = []
        self.result: SttResult = None

    def __call__(self, result: SttResult, options: Optional[dict] = None, **kwargs):
        return self.write(result=result, options=options, **kwargs)

    def write(self, result: SttResult, options: Optional[dict] = None, **kwargs):
        self.result = result

        self.output.seek(0)
        self.output.truncate(0)

        normalizer = SegmentNormalizer(
            segments=result.segments,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )

        total_duration = result.duration
        tqdm_bar = tqdm.tqdm(total=int(total_duration), desc="Transcribing")

        def _iter():
            try:
                for segment in normalizer.normalize(options=options, **kwargs):
                    self.subtitles.append(segment)
                    end = segment.end_s
                    next_n = int(end)
                    tqdm_bar.update(next_n - tqdm_bar.n)
                    yield segment
            finally:
                tqdm_bar.close()

        sub_segments = _iter()
        self.write_result(sub_segments, file=self.output, options=options, **kwargs)

        return self.get_result()

    def write_result(
        self,
        segments: Iterable[SubtitleSegment],
        file: TextIO,
        options: Optional[dict] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def get_result(self):
        self.output.seek(0)
        return self.output.read()


class WriteTXT(ResultWriter):
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self,
        segments: Iterable[SubtitleSegment],
        file: TextIO,
        options: Optional[dict] = None,
        **kwargs,
    ):
        for segment in segments:
            print(segment.text.strip(), file=file, flush=True)


class WriteVTT(ResultWriter):
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self,
        segments: Iterable[SubtitleSegment],
        file: TextIO,
        options: Optional[dict] = None,
        **kwargs,
    ):
        print("WEBVTT\n", file=file)
        for segment in segments:
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteSRT(ResultWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(
        self,
        segments: Iterable[SubtitleSegment],
        file: TextIO,
        options: Optional[dict] = None,
        **kwargs,
    ):
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteLRC(ResultWriter):
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self,
        segments: Iterable[SubtitleSegment],
        file: TextIO,
        options: Optional[dict] = None,
        **kwargs,
    ):
        for segment in segments:
            start = segment.start_s
            minutes = int(start // 60)
            seconds = start % 60
            timestamp = f"[{minutes:02}:{seconds:05.2f}]"
            text = segment.text.strip()
            print(f"{timestamp}{text}", file=file, flush=True)


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    def write_result(
        self,
        segments: Iterable[SubtitleSegment],
        file: TextIO,
        options: Optional[dict] = None,
        **kwargs,
    ):
        print("start", "end", "text", sep="\t", file=file)
        for segment in segments:
            start = segment.start_s
            end = segment.end_s
            text = segment.text.strip().replace("\t", " ")

            print(round(1000 * start), file=file, end="\t")
            print(round(1000 * end), file=file, end="\t")
            print(text, file=file, flush=True)


class WriteJSON(ResultWriter):
    always_include_hours: bool = True

    def write_result(
        self,
        segments: Iterable[SubtitleSegment],
        file: TextIO,
        options: Optional[dict] = None,
        **kwargs,
    ):
        def json_dump_default(o):
            if hasattr(o, "_asdict"):
                return o._asdict()
            # 判断是 dataclass
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if isinstance(o, dict):
                return o
            if isinstance(o, list):
                return o
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)

        json.dump(
            {
                "segments": list(segments),
                "language": self.result.language,
                "duration": self.result.duration,
            },
            file,
            ensure_ascii=False,
            default=json_dump_default,
        )


def get_writer(output_format: str) -> ResultWriter:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "lrc": WriteLRC,
        "json": WriteJSON,
    }

    return writers[output_format]()
