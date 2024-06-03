from modules.api.Api import APIManager

from modules.api.impl import (
    base_api,
    tts_api,
    ssml_api,
    google_api,
    openai_api,
    refiner_api,
)

api = APIManager()

base_api.setup(api)
tts_api.setup(api)
ssml_api.setup(api)
google_api.setup(api)
openai_api.setup(api)
refiner_api.setup(api)
