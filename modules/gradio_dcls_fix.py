def dcls_patch():
    from gradio import data_classes

    data_classes.PredictBody.__get_pydantic_json_schema__ = lambda x, y: {
        "type": "object",
    }
