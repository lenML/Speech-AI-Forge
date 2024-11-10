from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.data import styles_mgr


async def list_styles():
    return {"message": "ok", "data": styles_mgr.list_items()}


async def create_style():
    # TODO
    pass


def setup(app: APIManager):
    app.get("/v1/styles/list", response_model=api_utils.BaseResponse, tags=["Style"])(
        list_styles
    )
