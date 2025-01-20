import fnmatch
import logging

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse


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


CURRENT_CDN = "fastly.jsdelivr.net"

# 备选 CDN
# 节点                      描述            可用性
# gcore.jsdelivr.net	    Gcore 节点	    可用性高
# testingcf.jsdelivr.net	Cloudflare 节点	可用性高
# quantil.jsdelivr.net	    Quantil 节点	可用性一般
# fastly.jsdelivr.net	    Fastly 节点	    可用性一般
# originfastly.jsdelivr.net	Fastly 节点	    可用性低
# test1.jsdelivr.net	    Cloudflare 节点	可用性低
# cdn.jsdelivr.net	        通用节点	    可用性低


class CustomStaticFiles(StaticFiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if not isinstance(response, FileResponse):
            return response
        content_type = response.headers.get("Content-Type") or ""
        if "text/html" not in content_type:
            return response
        request = Request(scope)
        cdn_host = request.query_params.get("cdn_host")
        if cdn_host:
            response = self.replace_cdn_host(response, cdn_host)
        return response

    def replace_cdn_host(self, response: FileResponse, cdn_host: str):
        content_path = response.path
        with open(content_path, "rb") as f:
            content_bytes = f.read()
        content = content_bytes.decode("utf-8")
        content = content.replace(CURRENT_CDN, cdn_host)
        del response.headers["Content-Length"]
        new_response = Response(
            content, headers=response.headers, media_type=response.media_type
        )
        return new_response


class APIManager:
    def __init__(self, app: FastAPI, exclude_patterns=[]):
        self.app = app
        self.registered_apis = {}
        self.logger = logging.getLogger(__name__)
        self.exclude = exclude_patterns

        self.cors_enabled = False

    def is_excluded(self, path):
        return is_excluded(path, self.exclude)

    def set_cors(
        self,
        allow_origins: list = ["*"],
        allow_credentials: bool = True,
        allow_methods: list = ["*"],
        allow_headers: list = ["*"],
    ):
        if self.cors_enabled:
            raise Exception("CORS is already enabled")
        self.cors_enabled = True

        # NOTE: 为什么不用 CORSMiddleware ? 因为就是各种无效... 所以单独写了一个
        # 参考： https://github.com/fastapi/fastapi/issues/1663
        async def _set_cors_headers(response: Response, origin: str = None):
            """设置CORS响应头"""
            if origin:
                response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
            if allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"

        @self.app.middleware("http")
        async def cors_handler(request: Request, call_next):
            response: Response = await call_next(request)

            origin = request.headers.get("Origin")

            if "*" in allow_origins:
                if allow_credentials and origin:
                    # 当允许凭证时，需要匹配具体来源
                    await _set_cors_headers(response, origin)
                else:
                    # 否则直接允许所有来源
                    await _set_cors_headers(response, "*")
            elif origin and (origin in allow_origins):
                await _set_cors_headers(response, origin)

            # 处理预检请求
            if request.method == "OPTIONS":
                response.status_code = 200
                if not response.headers.get("Access-Control-Allow-Origin"):
                    if "*" in allow_origins:
                        await _set_cors_headers(response, "*")
                    else:
                        await _set_cors_headers(response, origin)
                else:
                    await _set_cors_headers(response)
            return response

    def setup_playground(self):
        app = self.app
        app.mount(
            "/playground",
            CustomStaticFiles(directory="playground", html=True),
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
