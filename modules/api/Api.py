from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

import logging

from fastapi.staticfiles import StaticFiles


class APIManager:
    def __init__(self, no_docs=False):
        self.app = FastAPI(
            title="ChatTTS Forge API",
            description="ChatTTS-Forge 是一个功能强大的文本转语音生成工具，支持通过类 SSML 语法生成丰富的音频长文本，并提供全面的 API 服务，适用于各种场景。\n\nChatTTS-Forge is a powerful text-to-speech generation tool that supports generating rich audio long texts through class SSML syntax\n\n https://github.com/lenML/ChatTTS-Forge",
            version="0.1.0",
            redoc_url=None if no_docs else "/redoc",
            docs_url=None if no_docs else "/docs",
        )
        self.registered_apis = {}
        self.logger = logging.getLogger(__name__)

    def set_cors(
        self,
        allow_origins: list = ["*"],
        allow_credentials: bool = True,
        allow_methods: list = ["*"],
        allow_headers: list = ["*"],
    ):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

    def setup_playground(self):
        app = self.app
        app.mount(
            "/playground",
            StaticFiles(directory="playground", html=True),
            name="playground",
        )

    def get(self, path: str, **kwargs):
        def decorator(func):
            self.app.get(path, **kwargs)(func)

            self.registered_apis[path] = func
            self.logger.info(f"Registered API: GET {path}")

            return func

        return decorator

    def post(self, path: str, **kwargs):
        def decorator(func):
            self.app.post(path, **kwargs)(func)

            self.registered_apis[path] = func
            self.logger.info(f"Registered API: POST {path}")

            return func

        return decorator
