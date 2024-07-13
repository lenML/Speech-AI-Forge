def dcls_patch():
    from gradio import data_classes

    # https://github.com/gradio-app/gradio/pull/8530
    data_classes.PredictBody.__get_pydantic_json_schema__ = lambda x, y: {
        "type": "object",
    }
