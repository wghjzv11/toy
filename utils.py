import re

def to_int(value: str) -> int:
    """
    '+57900'  -> 57900
    '-56000'  -> 56000
    '--275'   -> 275
    '53000'   -> 53000
    """
    return int(re.sub(r'^[+-]+', '', value.strip()))
