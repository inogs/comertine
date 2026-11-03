import re
from datetime import datetime
from datetime import timedelta

import numpy as np


def numpy_to_datetime(dt: np.datetime64) -> datetime:
    return dt.astype('datetime64[us]').astype(datetime)


# Units accepted (and produced) in the textual form of a duration, from the
# largest to the smallest
_TIMEDELTA_UNITS = {"d": "D", "h": "h", "m": "m", "s": "s"}

_TIMEDELTA_REGEX = re.compile(r"\s*(\d+)\s*([smhdSMHD])\s*")


def parse_timedelta(value: str) -> np.timedelta64:
    """
    Parse a duration like "120s", "30m", "8h" or "1d".

    Args:
        value: An integer followed by a unit (s, m, h or d).

    Returns:
        The corresponding np.timedelta64.

    Raises:
        ValueError: if `value` is not a valid duration.
    """
    match = _TIMEDELTA_REGEX.fullmatch(value)
    if match is None:
        msg = (
            f"invalid duration {value!r}: expected an integer followed by a "
            "unit (s, m, h or d), e.g. 120s, 30m, 8h"
        )
        raise ValueError(msg)
    amount, unit = match.groups()
    return np.timedelta64(int(amount), _TIMEDELTA_UNITS[unit.lower()])


def to_timedelta64(value: np.timedelta64 | timedelta | str) -> np.timedelta64:
    """
    Convert a duration into a np.timedelta64 with a resolution of seconds.

    Args:
        value: A np.timedelta64, a datetime.timedelta or a string accepted by
            `parse_timedelta`.

    Returns:
        A np.timedelta64 with unit "s".

    Raises:
        ValueError: if `value` is not a whole number of seconds.
        TypeError: if `value` has an unsupported type.
    """
    if isinstance(value, str):
        value = parse_timedelta(value)
    elif isinstance(value, timedelta):
        value = np.timedelta64(value)
    elif not isinstance(value, np.timedelta64):
        msg = f"cannot convert {value!r} (of type {type(value)}) to a duration"
        raise TypeError(msg)

    if np.isnat(value):
        msg = "a duration cannot be NaT"
        raise ValueError(msg)

    in_seconds = value.astype("timedelta64[s]")
    if in_seconds != value:
        msg = f"durations must be a whole number of seconds (received {value})"
        raise ValueError(msg)
    return in_seconds


def format_timedelta(value: np.timedelta64) -> str:
    """
    Write a duration in the format read by `parse_timedelta`.

    The largest unit that represents `value` exactly is used (e.g. "8h"
    instead of "28800s").
    """
    seconds = int(to_timedelta64(value) / np.timedelta64(1, "s"))
    for text_unit, numpy_unit in _TIMEDELTA_UNITS.items():
        unit_seconds = int(np.timedelta64(1, numpy_unit) / np.timedelta64(1, "s"))
        if seconds % unit_seconds == 0:
            return f"{seconds // unit_seconds}{text_unit}"
    raise AssertionError("unreachable: every duration is a multiple of 1s")
