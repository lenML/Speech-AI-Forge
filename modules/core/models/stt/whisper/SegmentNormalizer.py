import copy
from typing import Generator, Iterable, List, NamedTuple, Optional

from whisper.utils import format_timestamp

from modules.core.models.stt.whisper.whisper_dcls import SttSegment, SttWord


class SubtitleSegment(NamedTuple):
    start: str
    end: str
    text: str
    words: list
    start_s: float
    end_s: float


class SegmentNormalizer:

    def __init__(
        self,
        segments: Iterable[SttSegment],
        always_include_hours: bool,
        decimal_marker: str,
    ):
        self.segments = segments
        self.always_include_hours = always_include_hours
        self.decimal_marker = decimal_marker

    def normalize(
        self,
        options: Optional[dict] = None,
        *,
        max_line_width: Optional[int] = None,
        max_line_count: Optional[int] = None,
        max_words_per_line: Optional[int] = None,
    ) -> Generator[SubtitleSegment, None, None]:
        options = options or {}
        max_line_width = max_line_width or options.get("max_line_width")
        max_line_count = max_line_count or options.get("max_line_count")
        max_words_per_line = max_words_per_line or options.get("max_words_per_line")
        preserve_segments = max_line_count is None or max_line_width is None
        max_line_width = max_line_width or 1000
        max_words_per_line = max_words_per_line or 1000

        for subtitle in self._iterate_subtitles(
            max_line_width, max_line_count, preserve_segments, max_words_per_line
        ):
            subtitle_start = self._format_timestamp(subtitle[0].start)
            subtitle_end = self._format_timestamp(subtitle[-1].end)
            subtitle_text = "".join([word.word for word in subtitle])

            yield SubtitleSegment(
                start=subtitle_start,
                end=subtitle_end,
                text=subtitle_text,
                start_s=subtitle[0].start,
                end_s=subtitle[-1].end,
                # words=[w._asdict() for w in subtitle],
                words=subtitle,
            )

    def _iterate_subtitles(
        self,
        max_line_width: int,
        max_line_count: int,
        preserve_segments: bool,
        max_words_per_line: int,
    ):
        line_len = 0
        line_count = 1
        subtitle: List[SttWord] = []
        last: float = 0.0

        for segment in self.segments:
            chunk_index = 0
            words = segment.words
            if words is None or len(words) == 0:
                yield [SttWord(start=segment.start, end=segment.end, word=segment.text)]
                continue
            while chunk_index < len(words):
                words_count = min(max_words_per_line, len(words) - chunk_index)
                for i, original_timing in enumerate(
                    words[chunk_index : chunk_index + words_count]
                ):
                    timing: SttWord = copy.deepcopy(original_timing)
                    long_pause = not preserve_segments and timing.start - last > 3.0
                    has_room = line_len + len(timing.word) <= max_line_width
                    seg_break = i == 0 and len(subtitle) > 0 and preserve_segments

                    if line_len > 0 and has_room and not long_pause and not seg_break:
                        line_len += len(timing.word)
                    else:
                        # timing = timing._replace(word=timing.word.strip())
                        timing.word = timing.word.strip()
                        if (
                            len(subtitle) > 0
                            and max_line_count is not None
                            and (long_pause or line_count >= max_line_count)
                            or seg_break
                        ):
                            yield subtitle
                            subtitle = []
                            line_count = 1
                        elif line_len > 0:
                            line_count += 1
                            # timing = timing._replace(word="\n" + timing.word)
                            timing.word = "\n" + timing.word
                        line_len = len(timing.word.strip())
                    subtitle.append(timing)
                    last = timing.start
                chunk_index += words_count
        if len(subtitle) > 0:
            yield subtitle

    def _format_timestamp(self, seconds: float) -> str:
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )
