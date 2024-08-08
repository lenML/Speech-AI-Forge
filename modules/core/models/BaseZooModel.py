class BaseZooModel:

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.hash = ""

    def reset(self) -> None:
        """
        重置推理上下文
        """
        pass

    def is_loaded(self) -> bool:
        return False

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def check_exists(self) -> bool:
        return False

    def download(self) -> None:
        pass

    def interrupt(self) -> None:
        pass
