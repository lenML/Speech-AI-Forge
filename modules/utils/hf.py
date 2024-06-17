# 给huggingface space写的兼容代码

try:
    import spaces

    is_spaces_env = True
except:

    class NoneSpaces:
        def __init__(self):
            pass

        def GPU(self, *args, **kwargs):
            def _GPU(func):
                return func

            return _GPU

    spaces = NoneSpaces()
    is_spaces_env = False
