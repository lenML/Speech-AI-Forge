import os


def get_env_val(key, val_type):
    env_val = os.getenv(key.upper())
    if env_val is not None and env_val != "":
        if val_type == bool:
            return env_val.lower() in ("true", "1", "t")
        return val_type(env_val)

    if env_val == "":
        return None

    return env_val


def get_env_or_arg(args, arg_name, default, arg_type):
    arg_val = getattr(args, arg_name)
    env_val = get_env_val(arg_name, arg_type)

    if arg_type == bool and env_val is not None:
        return env_val

    if arg_val is not None:
        return arg_val
    elif env_val is not None:
        return env_val

    return default
