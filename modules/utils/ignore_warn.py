import warnings


def ignore_useless_warnings():

    # NOTE: 因为触发位置在 `vocos/heads.py:60` 改不动...所以忽略
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="ComplexHalf support is experimental"
    )
