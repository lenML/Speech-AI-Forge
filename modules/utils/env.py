import os


def get_env_or_arg(args, arg_name, default, arg_type):
    if getattr(args, arg_name) is not None:
        return getattr(args, arg_name)

    env_value = os.getenv(arg_name.upper())
    if env_value is not None and env_value != "":
        if arg_type == bool:
            return env_value.lower() in ("true", "1", "t")
        return arg_type(env_value)

    return default
