from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging

from fastapi.staticfiles import StaticFiles

import fnmatch


def is_excluded(path, exclude_patterns):
    """
    检查路径是否被排除

    :param path: 需要检查的路径
    :param exclude_patterns: 包含通配符的排除路径列表
    :return: 如果路径被排除，返回 True；否则返回 False
    """
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path, pattern):
            print(path, pattern)
            return True
    return False


class APIManager:
    def __init__(self, app: FastAPI, no_docs=False, exclude_patterns=[]):
        self.app = app
        self.registered_apis = {}
        self.logger = logging.getLogger(__name__)
        self.exclude = exclude_patterns

    def is_excluded(self, path):
        return is_excluded(path, self.exclude)

    def set_cors(
        self,
        allow_origins: list = ["*"],
        allow_credentials: bool = True,
        allow_methods: list = ["*"],
        allow_headers: list = ["*"],
    ):
        # reset middleware stack
        self.app.middleware_stack = None
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )
        self.app.build_middleware_stack()

    def setup_playground(self):
        app = self.app
        app.mount(
            "/playground",
            StaticFiles(directory="playground", html=True),
            name="playground",
        )

    def get(self, path: str, **kwargs):
        def decorator(func):
            if self.is_excluded(path):
                return func

            self.app.get(path, **kwargs)(func)

            self.registered_apis[path] = func
            self.logger.info(f"Registered API: GET {path}")

            return func

        return decorator

    def post(self, path: str, **kwargs):
        def decorator(func):
            if self.is_excluded(path):
                return func

            self.app.post(path, **kwargs)(func)

            self.registered_apis[path] = func
            self.logger.info(f"Registered API: POST {path}")

            return func

        return decorator
