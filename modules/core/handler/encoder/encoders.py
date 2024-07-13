from modules.core.handler.encoder.StreamEncoder import StreamEncoder


class Mp3Encoder(StreamEncoder):

    def open(self, acodec: str = "libmp3lame", bitrate: str = "128k"):
        return super().open("mp3", acodec, bitrate)


class WavEncoder(StreamEncoder):

    def open(self, acodec: str = "pcm_s16le", bitrate: str = "128k"):
        return super().open("wav", acodec, bitrate)


class OggEncoder(StreamEncoder):

    def open(self, acodec: str = "libvorbis", bitrate: str = "128k"):
        return super().open("ogg", acodec, bitrate)


class FlacEncoder(StreamEncoder):

    def open(self, acodec: str = "flac", bitrate: str = "128k"):
        return super().open("flac", acodec, bitrate)


class AacEncoder(StreamEncoder):

    def open(self, acodec: str = "aac", bitrate: str = "128k"):
        return super().open("aac", acodec, bitrate)
