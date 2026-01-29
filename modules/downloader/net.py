import os
import requests


def can_net_access(
    test_url="https://huggingface.co/",
    timeout=5,
):
    try:
        r = requests.get(test_url, timeout=timeout, allow_redirects=False)
        return r.status_code < 500
    except Exception as e:
        return False


if __name__ == "__main__":
    status = can_net_access()
    print(status)
