from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.data import styles_mgr


async def list_styles():
    return {"message": "ok", "data": styles_mgr.list_items()}


async def create_style():
    # TODO
    pass


def setup(app: APIManager):
    app.get(
        "/v1/styles/list",
        response_model=api_utils.BaseResponse,
        tags=["Style"],
        # 此 api 将废弃，取而代之的是 speaker 系统 和 prompt
        description="""
**DEPRECATED**
This API is deprecated and will be removed in the future. We will replace it with the speaker system and prompt system.
""",
    )(list_styles)
