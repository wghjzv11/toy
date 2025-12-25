def to_int(value: str) -> int:
    """
    '+57900'  -> 57900
    '-56000'  -> 56000
    '--275'   -> 275
    '53000'   -> 53000
    """
    value = value.strip()
    if value.startswith("-"):
        return int(value[1:])
    if value.startswith("--"):
        return int(value[2:])
    if value.startswith("+"):
        return int(value[1:])
    return int(value)