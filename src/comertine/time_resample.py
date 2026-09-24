from dataclasses import dataclass
from logging import getLogger

import dask.array as da
import numpy as np
import xarray as xr


LOGGER = getLogger(__name__)


@dataclass
class TimeFrameAssociation:
    time: np.datetime64
    associations: list[tuple[int, float]]


def resample_ds_in_time(ds, new_time_resolution: np.timedelta64):
    time = ds.time.values

    new_times = [TimeFrameAssociation(time=time[0], associations=[(0, 1.)])]
    while True:
        current_time = new_times[-1].time + new_time_resolution
        if current_time > time[-1]:
            break

        # Start index in the greatest timestep smaller or equal than the
        start_index = new_times[-1].associations[0][0]
        while start_index < len(time) - 1 and time[start_index] < current_time:
            start_index += 1

        if time[start_index] > current_time:
            start_index -= 1

        if time[start_index] == current_time:
            time_frame = TimeFrameAssociation(
                time=current_time,
                associations=[(start_index, 1.)]
            )
            new_times.append(time_frame)
            continue

        time_interval = time[start_index + 1] - time[start_index]
        current_position = (current_time - time[start_index]) / time_interval
        current_position = float(current_position)

        time_frame = TimeFrameAssociation(
            time=current_time,
            associations=[
                (start_index, 1 - current_position),
                (start_index + 1, current_position)
            ]
        )
        new_times.append(time_frame)

    new_vars = {}
    for d_var in ds.data_vars:
        if "time" not in ds[d_var].dims:
            new_vars[d_var] = ds[d_var]
            continue
        time_pos = ds[d_var].dims.index("time")
        previous_coords = dict(ds[d_var].coords)
        del previous_coords["time"]
        previous_coords["time"] = [t.time for t in new_times]

        def _get_slice(i: int) -> tuple[slice | int, ...]:
            slice_: list[slice | int] = [slice(None)] * len(ds[d_var].shape)
            slice_[time_pos] = i
            return tuple(slice_)

        slicing_db = [
            ds[d_var].data[_get_slice(i)] for i in range(len(ds[d_var].time))
        ]

        frames = []
        for time_frame in new_times:
            interpolated_frame = sum(
                slicing_db[s[0]] * s[1] for s in time_frame.associations
            )
            frames.append(interpolated_frame)
        new_data = da.stack(frames, axis=time_pos)

        new_d_var = xr.DataArray(
            new_data,
            dims=ds[d_var].dims,
            coords=previous_coords
        )
        new_vars[d_var] = new_d_var

    new_ds = xr.Dataset(new_vars)
    return new_ds


def extend_ds_in_time(ds, extension: np.timedelta64):
    """
    Extend a dataset in time by padding with its first and last frames.

    The "time" coordinate of `ds` is assumed to be uniformly spaced (as
    produced, for instance, by `resample_ds_in_time`). The dataset is
    extended by `extension` in both directions: the value at `time[0]` is
    repeated backwards and the value at `time[-1]` is repeated forwards, so
    that the returned dataset covers at least the interval
    [time[0] - extension, time[-1] + extension].

    Args:
        ds: the dataset to extend
        extension: how much the dataset must be extended in each direction

    Returns:
        A new dataset extended in time.
    """
    time = ds.time.values
    time_resolution = time[1] - time[0]

    n_steps = int(np.ceil(extension / time_resolution))

    new_time = np.concatenate([
        time[0] + np.arange(-n_steps, 0) * time_resolution,
        time,
        time[-1] + np.arange(1, n_steps + 1) * time_resolution,
    ])

    new_vars = {}
    for d_var in ds.data_vars:
        if "time" not in ds[d_var].dims:
            new_vars[d_var] = ds[d_var]
            continue

        time_pos = ds[d_var].dims.index("time")
        data = ds[d_var].data

        def _get_frame_slice(i: int) -> tuple[slice, ...]:
            slice_: list[slice] = [slice(None)] * data.ndim
            slice_[time_pos] = slice(i, i + 1)
            return tuple(slice_)

        first_frame = data[_get_frame_slice(0)]
        last_frame = data[_get_frame_slice(data.shape[time_pos] - 1)]

        first_padding = da.repeat(first_frame, n_steps, axis=time_pos)
        last_padding = da.repeat(last_frame, n_steps, axis=time_pos)

        new_data = da.concatenate(
            [first_padding, data, last_padding], axis=time_pos
        )

        previous_coords = dict(ds[d_var].coords)
        del previous_coords["time"]
        previous_coords["time"] = new_time

        new_vars[d_var] = xr.DataArray(
            new_data, dims=ds[d_var].dims, coords=previous_coords
        )

    return xr.Dataset(new_vars)
