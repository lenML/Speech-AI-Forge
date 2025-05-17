from urllib.parse import urlencode, urljoin

import requests


class SAFClient:
    def __init__(self, base_url="http://localhost:7870/"):
        self.base_url = base_url

    def _join(self, path, querys=None):
        """
        拼接 URL，包含查询参数。

        :param path: URL 路径
        :param querys: 查询参数字典
        :return: 完整的 URL
        """
        url = urljoin(self.base_url, path)
        if querys:
            url = f"{url}?{urlencode(querys)}"
        return url

    def _request(self, path: str, body=None, method=None):
        """
        发送 HTTP 请求。

        :param path: API 路径
        :param body: 请求体（JSON 数据或查询参数）
        :param method: HTTP 方法，默认为 GET 或根据 body 自动判断
        :return: HTTP 响应
        """
        method = method or ("POST" if body else "GET")
        url = self._join(path, body if method == "GET" else None)
        headers = (
            {"Content-Type": "application/json"} if body and method != "GET" else {}
        )

        response = requests.request(
            method, url, json=body if method != "GET" else None, headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Unexpected status code {response.status_code}")
        return response

    def ping(self):
        """
        检查服务可用性。

        :return: 服务响应时间（毫秒）
        """
        import time

        start = time.time()
        self._request("/v1/ping")
        return {"time": int((time.time() - start) * 1000)}

    def list_models(self):
        """
        获取可用模型列表。

        :return: 模型列表
        """
        response = self._request("/v1/models/list")
        data = response.json().get("data", [])
        return {"models": data}

    def tts(self, params):
        """
        获取文本转语音结果。

        :param params: TTS 参数
        :return: 音频数据（二进制）
        """
        response = self._request("/v1/tts", params, method="GET")
        return response.content
