from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.models import reload_chat_tts


def setup(app: APIManager):
    @app.get("/v1/models/reload", response_model=api_utils.BaseResponse)
    async def reload_models():
        # Reload models
        reload_chat_tts()
        return api_utils.success_response("Models reloaded")
