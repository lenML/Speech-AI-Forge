from modules.core.handler.encoder.StreamEncoder import StreamEncoder


class Mp3Encoder(StreamEncoder):

    def open(self, acodec: str = "libmp3lame", bitrate: str = "128k"):
        acodec = acodec or "libmp3lame"
        return super().open("mp3", acodec, bitrate)


class WavEncoder(StreamEncoder):

    def open(self, acodec: str = "pcm_s16le", bitrate: str = "128k"):
        acodec = acodec or "pcm_s16le"
        return super().open("wav", acodec, bitrate)


class OggEncoder(StreamEncoder):

    def open(self, acodec: str = "libvorbis", bitrate: str = "128k"):
        acodec = acodec or "libvorbis"
        return super().open("ogg", acodec, bitrate)


class FlacEncoder(StreamEncoder):

    def open(self, acodec: str = "flac", bitrate: str = "128k"):
        acodec = acodec or "flac"
        return super().open("flac", acodec, bitrate)


class AacEncoder(StreamEncoder):

    def open(self, acodec: str = "aac", bitrate: str = "128k"):
        acodec = acodec or "acc"
        return super().open("aac", acodec, bitrate)


class RawEncoder(StreamEncoder):

    def __init__(self) -> None:
        super().__init__()

        # FIXME: This is a hack to make the encoder work with the current implementation of the StreamEncoder.
        self.p = True

    def open(self, format: str = "", acodec: str = "", bitrate: str = ""):
        pass

    def write(self, data: bytes):
        self.output_queue.put(data)

    def close(self):
        pass

    def terminate(self):
        pass
