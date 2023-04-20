from __future__ import annotations

import time


def get_time_as_string() -> str:
    """Gets the current date and time as a string."""
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return time_str
