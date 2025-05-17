import json

from modules.api.worker import app

if __name__ == "__main__":
    """
    这个脚本将启动 fastapi 服务，并获取 openapi.json 保存到 "./docs/openapi.json" 位置
    """
    with open("./docs/openapi.json", "w") as f:
        spec_json = app.openapi()
        # 添加 servers
        spec_json["servers"] = [
            {"url": "http://127.0.0.1:7870"},
            {"url": "http://0.0.0.0:7870"},
            {"url": "http://localhost:7870"},
        ]
        # 开启 indent 是为了方便 git 生成 diff，某些时候可以用来观察 api 变化
        json.dump(app.openapi(), f, indent=2, ensure_ascii=False)
    print("openapi.json saved to ./docs/openapi.json")
