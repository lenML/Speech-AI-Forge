from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

import logging

from fastapi.staticfiles import StaticFiles


class APIManager:
    def __init__(self):
        self.app = FastAPI()
        self.registered_apis = {}
        self.logger = logging.getLogger(__name__)

        self.setup_static()

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

    def setup_static(self):
        app = self.app
        app.mount("/playground", StaticFiles(directory="playground"), name="playground")

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
