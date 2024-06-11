# 给huggingface space写的兼容代码

try:
    import spaces

    is_spaces_env = True
except:

    class NoneSpaces:
        def __init__(self):
            pass

        def GPU(self, fn):
            return fn

    spaces = NoneSpaces()
    is_spaces_env = False
