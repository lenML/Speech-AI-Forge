def to_number(value, t, default=0):
    try:
        number = t(value)
        return number
    except (ValueError, TypeError) as e:
        return default
