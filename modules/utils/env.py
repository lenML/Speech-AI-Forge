import os


def get_env_or_arg(args, arg_name, default, arg_type):
    # 检查命令行参数中是否显式设置了该参数
    if getattr(args, arg_name) is not None:
        return getattr(args, arg_name)

    # 如果命令行参数未显式设置，则使用环境变量中的值
    env_value = os.getenv(arg_name.upper())
    if env_value is not None:
        if arg_type == bool:
            return env_value.lower() in ("true", "1", "t")
        return arg_type(env_value)

    # 如果环境变量也没有设置，则使用默认值
    return default
