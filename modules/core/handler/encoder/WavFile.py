import io
import logging
import struct


class WAVFileBytes:

    def __init__(self, wav_bytes):
        self.wav_bytes = wav_bytes
        self.riff = None
        self.size = None
        self.fformat = None
        self.aformat = None
        self.channels = None
        self.samplerate = None
        self.bitrate = None
        self.subchunks = []
        self.header_end = 0
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def read(self):
        with io.BytesIO(self.wav_bytes) as fh:
            self.riff, self.size, self.fformat = struct.unpack("<4sI4s", fh.read(12))
            logging.info(
                "Riff: %s, Chunk Size: %i, format: %s",
                self.riff,
                self.size,
                self.fformat,
            )

            # Read header
            chunk_header = fh.read(8)
            subchunkid, subchunksize = struct.unpack("<4sI", chunk_header)

            if subchunkid == b"fmt ":
                (
                    self.aformat,
                    self.channels,
                    self.samplerate,
                    byterate,
                    blockalign,
                    bps,
                ) = struct.unpack("HHIIHH", fh.read(16))
                self.bitrate = (self.samplerate * self.channels * bps) / 1024
                logging.info(
                    "Format: %i, Channels %i, Sample Rate: %i, Kbps: %i",
                    self.aformat,
                    self.channels,
                    self.samplerate,
                    self.bitrate,
                )

            chunkOffset = fh.tell()
            while chunkOffset < self.size:
                fh.seek(chunkOffset)
                subchunk2id, subchunk2size = struct.unpack("<4sI", fh.read(8))
                logging.info("chunk id: %s, size: %i", subchunk2id, subchunk2size)
                subchunk_data = {"id": subchunk2id, "size": subchunk2size}

                if subchunk2id == b"LIST":
                    listtype = struct.unpack("<4s", fh.read(4))
                    subchunk_data["listtype"] = listtype
                    logging.info(
                        "\tList Type: %s, List Size: %i", listtype, subchunk2size
                    )

                    listOffset = 0
                    list_items = []
                    while (subchunk2size - 8) >= listOffset:
                        listitemid, listitemsize = struct.unpack("<4sI", fh.read(8))
                        listOffset = listOffset + listitemsize + 8
                        listdata = fh.read(listitemsize)
                        list_items.append(
                            {
                                "id": listitemid.decode("ascii"),
                                "size": listitemsize,
                                "data": listdata.decode("ascii"),
                            }
                        )
                        logging.info(
                            "\tList id %s, size: %i, data: %s",
                            listitemid.decode("ascii"),
                            listitemsize,
                            listdata.decode("ascii"),
                        )
                        logging.info("\tOffset: %i", listOffset)
                    subchunk_data["items"] = list_items
                elif subchunk2id == b"data":
                    logging.info("Found data")
                else:
                    subchunk_data["data"] = fh.read(subchunk2size).decode("ascii")
                    logging.info("Data: %s", subchunk_data["data"])

                self.subchunks.append(subchunk_data)
                chunkOffset = chunkOffset + subchunk2size + 8

            self.header_end = fh.tell()

    def get_header_data(self):
        return self.wav_bytes[: self.header_end]

    def get_body_data(self):
        return self.wav_bytes[self.header_end :]
