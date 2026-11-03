from collections.abc import Callable
from logging import getLogger
from typing import cast
from typing import overload

import numpy as np
import xarray as xr


LOGGER = getLogger(__name__)

DEFAULT_MAX_SOURCE_FRAMES = 128

WindowFunction = Callable[[np.ndarray, np.ndarray], list[tuple[slice, slice]]]


def regular_times(
    start_time: np.datetime64,
    end_time: np.datetime64,
    time_step: np.timedelta64,
) -> np.ndarray:
    """
    Generate equally spaced times from `start_time` up to `end_time`.

    The first element is exactly `start_time`; the following ones are
    obtained by adding `time_step` until `end_time` is exceeded. `end_time`
    is included only if it is reached exactly.

    Args:
        start_time: The first time of the output array.
        end_time: The upper bound (inclusive) of the output array.
        time_step: The (positive) distance between two consecutive times.

    Returns:
        A one-dimensional array of datetime64 values.
    """
    time_step = np.timedelta64(time_step)
    if time_step <= np.timedelta64(0, "s"):
        msg = f"time_step must be positive (received {time_step})"
        raise ValueError(msg)

    if end_time < start_time:
        msg = (
            f"end_time ({end_time}) must not precede start_time "
            f"({start_time})"
        )
        raise ValueError(msg)

    n_steps = int((end_time - start_time) // time_step)
    return start_time + np.arange(n_steps + 1) * time_step


def _validate_time_axes(
    source_times: np.ndarray, target_times: np.ndarray
) -> None:
    if source_times.ndim != 1 or target_times.ndim != 1:
        msg = "source_times and target_times must be one-dimensional"
        raise ValueError(msg)

    if np.any(source_times[1:] < source_times[:-1]):
        msg = "source_times must be sorted in increasing order"
        raise ValueError(msg)

    if np.any(target_times[1:] < target_times[:-1]):
        msg = "target_times must be sorted in increasing order"
        raise ValueError(msg)


def _time_window_slices(
    source_times: np.ndarray,
    target_times: np.ndarray,
    max_source_frames: int,
) -> list[tuple[slice, slice]]:
    """
    Plan interpolation windows with a bounded number of source frames.

    To resample a dataset efficiently, the source times are split into
    windows of at most `max_source_frames` frames. Each element of the
    returned list is a pair `(source_slice, target_slice)`: the target
    times selected by `target_slice` can be linearly interpolated using only
    the source frames selected by `source_slice`.

    Consecutive windows share one source frame, so the interpolation stays
    continuous at the edges. The number of target times per window depends
    on the spacing of the source times.

    Args:
        source_times: Monotonically increasing array of source times.
        target_times: Monotonically increasing array of target times.
        max_source_frames: Maximum number of source frames per window. Must
            be at least 2 to allow for interpolation.

    Returns:
        A list of `(source_slice, target_slice)` tuples.
    """
    if max_source_frames <= 1:
        msg = "max_source_frames must be at least 2"
        raise ValueError(msg)

    _validate_time_axes(source_times, target_times)

    if source_times.size <= 1 or target_times.size == 0:
        return [(slice(0, source_times.size), slice(0, target_times.size))]

    last_source = source_times.size - 1

    # Window k spans source frames [k * step, min((k + 1) * step, last)]
    step = max_source_frames - 1
    source_starts = np.arange(0, last_source, step)
    source_stops = np.minimum(source_starts + step, last_source)

    # The first window also takes the targets before source_times[0] and the
    # last one the targets after source_times[-1] (they are filled later
    # with the boundary values).
    target_starts = np.searchsorted(
        target_times, source_times[source_starts], side="left"
    )
    target_starts[0] = 0
    target_stops = np.searchsorted(
        target_times, source_times[source_stops], side="left"
    )
    target_stops[-1] = target_times.size

    return [
        (slice(int(s_start), int(s_stop) + 1), slice(int(t_start), int(t_stop)))
        for s_start, s_stop, t_start, t_stop in zip(
            source_starts, source_stops, target_starts, target_stops,
            strict=True,
        )
        if t_start < t_stop
    ]


def _time_window_slices_fixed_output(
    source_times: np.ndarray,
    target_times: np.ndarray,
    max_target_frames: int,
) -> list[tuple[slice, slice]]:
    """
    Plan interpolation windows with a bounded number of target frames.

    Like `_time_window_slices`, but instead of limiting the number of source
    frames per window, it limits the number of target frames: each window
    covers at most `max_target_frames` target times, while its source frames
    are chosen to bracket that target range.

    Args:
        source_times: Monotonically increasing array of source times.
        target_times: Monotonically increasing array of target times.
        max_target_frames: Maximum number of target frames per window.

    Returns:
        A list of `(source_slice, target_slice)` tuples.
    """
    if max_target_frames <= 0:
        msg = "max_target_frames must be at least 1"
        raise ValueError(msg)

    _validate_time_axes(source_times, target_times)

    if source_times.size <= 1 or target_times.size == 0:
        return [(slice(0, source_times.size), slice(0, target_times.size))]

    target_starts = np.arange(0, target_times.size, max_target_frames)
    target_stops = np.minimum(
        target_starts + max_target_frames, target_times.size
    )

    # floor / ceiling searchsorted to bracket the full target sub-range
    source_starts = np.maximum(
        np.searchsorted(source_times, target_times[target_starts], "right") - 1,
        0,
    )
    source_stops = np.minimum(
        np.searchsorted(source_times, target_times[target_stops - 1], "left"),
        source_times.size - 1,
    )

    return [
        (slice(int(s_start), int(s_stop) + 1), slice(int(t_start), int(t_stop)))
        for s_start, s_stop, t_start, t_stop in zip(
            source_starts, source_stops, target_starts, target_stops,
            strict=True,
        )
    ]


def _resample_internal(
    data: xr.Dataset | xr.DataArray,
    target_times: np.ndarray,
    window_fn: WindowFunction,
    *,
    allowed_outside: np.timedelta64 | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Core implementation shared by all the resample functions.

    Validates the time bounds, delegates the windowing to `window_fn`,
    performs a piecewise linear interpolation window by window and fills the
    target times outside the source range with the nearest boundary value of
    `data` (instead of extrapolating).

    Args:
        data: The dataset or array to reinterpolate.
        target_times: The (increasing) times of the output.
        window_fn: Callable `(source_times, target_times) -> windows` that
            partitions the time axes into `(source_slice, target_slice)`
            pairs driving the chunked interpolation.
        allowed_outside: Maximum allowed distance that the target times can
            be outside the data time range. Defaults to zero (strict).

    Returns:
        A dataset or array defined on `target_times`.
    """
    if allowed_outside is None:
        allowed_outside = np.timedelta64(0, "s")

    allowed_outside = np.timedelta64(allowed_outside)
    if allowed_outside < np.timedelta64(0, "s"):
        msg = "allowed_outside must be non-negative"
        raise ValueError(msg)

    target_times = np.asarray(target_times)
    if target_times.size == 0:
        msg = "target_times must contain at least one time value"
        raise ValueError(msg)

    source_times = np.asarray(data.time.values)
    if source_times.size == 0:
        msg = "data must contain at least one time value"
        raise ValueError(msg)

    first_source_time = source_times[0]
    last_source_time = source_times[-1]

    zero = np.timedelta64(0, "s")
    lower_outside = max(first_source_time - target_times[0], zero)
    upper_outside = max(target_times[-1] - last_source_time, zero)

    if lower_outside > allowed_outside or upper_outside > allowed_outside:
        msg = (
            "Requested interval exceeds allowed outside range: "
            f"before={lower_outside}, after={upper_outside}, "
            f"allowed={allowed_outside}"
        )
        raise ValueError(msg)

    windows = window_fn(source_times, target_times)

    if len(windows) == 1:
        # skip isel overhead when all data fits in one pass
        reinterpolated = data.interp(time=target_times)
    else:
        reinterpolated_chunks = [
            data.isel(time=source_slice).interp(
                time=target_times[target_slice]
            )
            for source_slice, target_slice in windows
        ]
        reinterpolated = cast(
            "xr.Dataset | xr.DataArray",
            xr.concat(
                reinterpolated_chunks,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                join="override",
            ),
        )

    time_coord = reinterpolated.time.values

    before_mask = target_times < first_source_time
    if np.any(before_mask):
        before = xr.DataArray(
            before_mask, dims=("time",), coords={"time": time_coord}
        )
        reinterpolated = _fill_where(
            reinterpolated, before, data.isel(time=0, drop=True)
        )

    after_mask = target_times > last_source_time
    if np.any(after_mask):
        after = xr.DataArray(
            after_mask, dims=("time",), coords={"time": time_coord}
        )
        reinterpolated = _fill_where(
            reinterpolated, after, data.isel(time=-1, drop=True)
        )

    return reinterpolated


def _fill_where(
    data: xr.Dataset | xr.DataArray,
    mask: xr.DataArray,
    fill_value: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """
    Replace the values of `data` with `fill_value` where `mask` is True.

    Only the variables that depend on time are modified: applying `where`
    to the whole dataset would broadcast the time-independent variables
    along the time dimension.
    """
    if isinstance(data, xr.DataArray):
        return data.where(~mask, fill_value)

    data = data.copy()
    for var_name, var in data.data_vars.items():
        if "time" in var.dims:
            data[var_name] = var.where(~mask, fill_value[var_name])
    return data


@overload
def resample_in_time(
    data: xr.DataArray,
    target_times: np.ndarray,
    *,
    max_source_frames: int = DEFAULT_MAX_SOURCE_FRAMES,
    allowed_outside: np.timedelta64 | None = None,
) -> xr.DataArray: ...


@overload
def resample_in_time(
    data: xr.Dataset,
    target_times: np.ndarray,
    *,
    max_source_frames: int = DEFAULT_MAX_SOURCE_FRAMES,
    allowed_outside: np.timedelta64 | None = None,
) -> xr.Dataset: ...


def resample_in_time(
    data: xr.Dataset | xr.DataArray,
    target_times: np.ndarray,
    *,
    max_source_frames: int = DEFAULT_MAX_SOURCE_FRAMES,
    allowed_outside: np.timedelta64 | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Linearly reinterpolate the time dimension of `data` on `target_times`.

    The work is split into windows of at most `max_source_frames` source
    frames, so that (when `data` is backed by dask) only a bounded number of
    frames is involved in each interpolation.

    Target times that fall before or after the source range are filled with
    the nearest boundary value rather than extrapolated, up to the tolerance
    set by `allowed_outside`.

    Args:
        data: The dataset or array to reinterpolate.
        target_times: The (increasing) times of the output.
        max_source_frames: Maximum number of source time frames per
            interpolation window.
        allowed_outside: Maximum allowed distance that the target times can
            be outside the data time range. Defaults to zero.

    Returns:
        A dataset or array defined on `target_times`.
    """
    return _resample_internal(
        data,
        target_times,
        lambda st, tt: _time_window_slices(
            st, tt, max_source_frames=max_source_frames
        ),
        allowed_outside=allowed_outside,
    )


@overload
def resample_in_time_fixed_chunks(
    data: xr.DataArray,
    target_times: np.ndarray,
    *,
    output_chunk_size: int = DEFAULT_MAX_SOURCE_FRAMES,
    allowed_outside: np.timedelta64 | None = None,
) -> xr.DataArray: ...


@overload
def resample_in_time_fixed_chunks(
    data: xr.Dataset,
    target_times: np.ndarray,
    *,
    output_chunk_size: int = DEFAULT_MAX_SOURCE_FRAMES,
    allowed_outside: np.timedelta64 | None = None,
) -> xr.Dataset: ...


def resample_in_time_fixed_chunks(
    data: xr.Dataset | xr.DataArray,
    target_times: np.ndarray,
    *,
    output_chunk_size: int = DEFAULT_MAX_SOURCE_FRAMES,
    allowed_outside: np.timedelta64 | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Linearly reinterpolate the time dimension of `data` on `target_times`.

    Like `resample_in_time`, but chunks the work by a fixed number of
    *output* (target) time frames rather than by source frames. This is
    useful when the desired output chunk size is known in advance, e.g. when
    writing to a chunked store.

    Args:
        data: The dataset or array to reinterpolate.
        target_times: The (increasing) times of the output.
        output_chunk_size: Maximum number of output time frames per
            interpolation window.
        allowed_outside: Maximum allowed distance that the target times can
            be outside the data time range. Defaults to zero.

    Returns:
        A dataset or array defined on `target_times`.
    """
    return _resample_internal(
        data,
        target_times,
        lambda st, tt: _time_window_slices_fixed_output(
            st, tt, max_target_frames=output_chunk_size
        ),
        allowed_outside=allowed_outside,
    )


def resample_ds_in_time(
    ds: xr.Dataset,
    new_time_resolution: np.timedelta64,
    *,
    extension: np.timedelta64 | None = None,
    max_source_frames: int = DEFAULT_MAX_SOURCE_FRAMES,
) -> xr.Dataset:
    """
    Resample a dataset on equally spaced times.

    The new times are spaced by `new_time_resolution` and start from the
    first time of `ds`; the last one is the greatest that does not exceed
    the last time of `ds`. Values are linearly interpolated in time.

    If `extension` is given, the new times are extended in both directions
    by the smallest multiple of `new_time_resolution` that is at least
    `extension`; outside the time range of `ds` the first (or last) frame of
    `ds` is repeated.

    Args:
        ds: The dataset to resample.
        new_time_resolution: The distance between two consecutive times of
            the output.
        extension: How much the output must be extended (in each direction)
            beyond the time range of `ds`. Defaults to no extension.
        max_source_frames: Maximum number of source time frames per
            interpolation window (see `resample_in_time`).

    Returns:
        A new dataset defined on the new times.
    """
    new_time_resolution = np.timedelta64(new_time_resolution)

    padding = np.timedelta64(0, "s")
    if extension is not None:
        extension = np.timedelta64(extension)
        if extension < np.timedelta64(0, "s"):
            msg = f"extension must be non-negative (received {extension})"
            raise ValueError(msg)
        n_steps = int(np.ceil(extension / new_time_resolution))
        padding = n_steps * new_time_resolution

    time = ds.time.values
    new_times = regular_times(
        time[0] - padding, time[-1] + padding, new_time_resolution
    )
    return resample_in_time(
        ds,
        new_times,
        max_source_frames=max_source_frames,
        allowed_outside=padding,
    )
