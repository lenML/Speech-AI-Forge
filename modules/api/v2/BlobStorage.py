import io
import threading
import time
from typing import Dict


# NOTE: 保存/定位 数据流
# 目前没有删除功能，每个 blob 会有一个过期时间，会定时清理到期的 blob
class Blob:
    def __init__(self, ttl: int) -> None:
        self.data = io.BytesIO()
        self.expiration_time = time.time() + ttl

    def write(self, data: bytes) -> None:
        self.data.write(data)

    def read(self) -> bytes:
        self.data.seek(0)
        return self.data.read()

    def is_expired(self) -> bool:
        return time.time() > self.expiration_time

    def __repr__(self) -> str:
        return f"<Blob(expired={self.is_expired()})>"


class BlobStorage:

    def __init__(self, cleanup_interval: int = 60) -> None:
        self.blobs: Dict[str, Blob] = {}
        self.cleanup_interval = cleanup_interval
        self._start_cleanup()

    def create_blob(self, key: str, ttl: int) -> Blob:
        blob = Blob(ttl=ttl)
        self.blobs[key] = blob
        return blob

    def get_blob(self, key: str) -> Blob:
        blob = self.blobs.get(key)
        if blob and not blob.is_expired():
            return blob
        elif blob:
            del self.blobs[key]
            raise KeyError("Blob has expired.")
        raise KeyError("Blob not found.")

    def clean_expired_blobs(self) -> None:
        keys_to_delete = [key for key, blob in self.blobs.items() if blob.is_expired()]
        for key in keys_to_delete:
            del self.blobs[key]
        print("Expired blobs cleaned up.")

    def _start_cleanup(self):
        """
        Starts the periodic cleanup task using threading.Timer.
        """
        self._cleanup_timer = threading.Timer(self.cleanup_interval, self._run_cleanup)
        self._cleanup_timer.start()

    def _run_cleanup(self):
        """
        Runs the cleanup and reschedules the next run.
        """
        self.clean_expired_blobs()
        self._start_cleanup()  # Reschedule the next cleanup

    def __del__(self):
        # Cancel the cleanup timer when BlobStorage is deleted
        if hasattr(self, "_cleanup_timer"):
            self._cleanup_timer.cancel()

    def __repr__(self) -> str:
        return f"BlobStorage({len(self.blobs)} blobs)"
