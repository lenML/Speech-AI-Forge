from modules.api.worker import app
import json

if __name__ == "__main__":
    """
    这个脚本将启动 fastapi 服务，并获取 openapi.json 保存到 "./docs/openapi.json" 位置
    """
    with open("./docs/openapi.json", "w") as f:
        json.dump(app.openapi(), f, ensure_ascii=False)
    print("openapi.json saved to ./docs/openapi.json")
