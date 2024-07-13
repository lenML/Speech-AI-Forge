from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.Enhancer.ResembleEnhance import reload_enhancer, unload_enhancer
from modules.core.models import zoo


def setup(app: APIManager):
    """
    TODO: 需要增强 zoo 以支持多 models 管理
    """

    @app.get("/v1/models/reload", response_model=api_utils.BaseResponse)
    async def reload_models():
        zoo.ChatTTS.reload_chat_tts()
        reload_enhancer()
        return api_utils.success_response("Models reloaded")

    @app.get("/v1/models/unload", response_model=api_utils.BaseResponse)
    async def reload_models():
        zoo.ChatTTS.unload_chat_tts()
        unload_enhancer()
        return api_utils.success_response("Models unloaded")
