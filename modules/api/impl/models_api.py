from modules.Enhancer.ResembleEnhance import reload_enhancer, unload_enhancer
from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.models import reload_chat_tts, unload_chat_tts


def setup(app: APIManager):
    @app.get("/v1/models/reload", response_model=api_utils.BaseResponse)
    async def reload_models():
        reload_chat_tts()
        reload_enhancer()
        return api_utils.success_response("Models reloaded")

    @app.get("/v1/models/unload", response_model=api_utils.BaseResponse)
    async def reload_models():
        unload_chat_tts()
        unload_enhancer()
        return api_utils.success_response("Models unloaded")
