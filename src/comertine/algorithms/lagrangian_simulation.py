from datetime import datetime
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from pydantic import BaseModel

from comertine.algorithms.explicit_euler import evolve_particles
from comertine.algorithms.initializers import initialize_particles
from comertine.domains import Domain
from comertine.tools.dask_tools import run_tasks
from comertine.tools.file_names import get_data_file_path
from comertine.tools.geometry import Grid
from comertine.tools.particle_config import ParticleConfig
from comertine.tools.time_conversions import numpy_to_datetime
from comertine.tools.time_resample import regular_times


class LagrangianSimulation(BaseModel):
    domain: Domain
    date: datetime
    config: ParticleConfig


def _n_steps(config: ParticleConfig) -> int:
    """
    The number of time steps that a batch performs.

    A batch saves its state at the times `start + i * time_interval` for
    `i = 0, ..., n_steps`, i.e. until at least `batch_time_length` has
    elapsed.
    """
    return int(np.ceil(config.batch_time_length / config.time_interval))


def batch_start_times(
    source_times: np.ndarray, config: ParticleConfig
) -> np.ndarray:
    """
    Compute the times when the batches of particles are spawned.

    The spawning starts `batch_time_length` (rounded up to a multiple of
    `time_interval`) before the first time of the data, so that, since the
    very beginning of the data, there are particles of all the ages. For the
    same reason, it continues until the same amount of time after the end of
    the data. Outside the time range of the data, the velocity field is
    frozen to its first (or last) frame.

    Args:
        source_times: The times of the velocity dataset.
        config: The configuration of the simulation.

    Returns:
        An array with the starting times of the batches.
    """
    padding = _n_steps(config) * config.time_interval

    # The last time of the data reachable with steps of `time_interval`
    last_time = regular_times(
        source_times[0], source_times[-1], config.time_interval
    )[-1]

    return regular_times(
        source_times[0] - padding, last_time + padding, config.batch_every
    )


def _source_window(
    source_times: np.ndarray, start_time: np.datetime64, end_time: np.datetime64
) -> slice:
    """
    Select the frames of the data needed to interpolate in [start, end].

    The interval is clipped to the time range of the data: outside of it,
    the first (or last) frame is used.
    """
    start_time = np.clip(start_time, source_times[0], source_times[-1])
    end_time = np.clip(end_time, source_times[0], source_times[-1])

    first = int(np.searchsorted(source_times, start_time, side="right")) - 1
    last = int(np.searchsorted(source_times, end_time, side="left"))

    first = max(first, 0)
    last = min(last, source_times.size - 1)
    return slice(first, last + 1)


def _interpolate_frame(
    frames: np.ndarray, frame_times: np.ndarray, time: np.datetime64
) -> np.ndarray:
    """
    Linearly interpolate in time a stack of frames (time is the first axis).

    Times outside `frame_times` get the first (or the last) frame.
    """
    if frame_times.size == 1:
        return frames[0]

    time = np.clip(time, frame_times[0], frame_times[-1])
    i = int(np.searchsorted(frame_times, time, side="right")) - 1
    i = min(max(i, 0), frame_times.size - 2)

    weight = float((time - frame_times[i]) / (frame_times[i + 1] - frame_times[i]))
    if weight == 0.:
        return frames[i]
    if weight == 1.:
        return frames[i + 1]
    return (1. - weight) * frames[i] + weight * frames[i + 1]


def run_batch(
    uo: np.ndarray,
    vo: np.ndarray,
    frame_times: np.ndarray,
    grid: Grid,
    batch_index: int,
    batch_start_time: np.datetime64,
    output_dir: Path,
    config: ParticleConfig,
    seed: np.random.SeedSequence,
) -> Path:
    """
    Spawn a batch of particles and evolve it for `config.batch_time_length`.

    Args:
        uo: The eastward velocity on the frames needed by the batch; the
            first axis is time. While moving the particles, it is multiplied
            by `config.increased_speed_factor`.
        vo: The northward velocity, like `uo`.
        frame_times: The times of the frames of `uo` and `vo`.
        grid: The grid of the model.
        batch_index: The index of the batch (used for the output file name).
        batch_start_time: When the particles are spawned.
        output_dir: Where the output file will be written.
        config: The configuration of the simulation.
        seed: The seed of the random generator of this batch.

    Returns:
        The path of the file with the trajectories of the particles.
    """
    random_generator = np.random.default_rng(seed)
    dt_seconds = config.time_interval_seconds

    # The data are usually float32: interpolating in time in float32 would
    # lose precision
    uo = np.asarray(uo, dtype=np.float64)
    vo = np.asarray(vo, dtype=np.float64)

    times = batch_start_time + np.arange(_n_steps(config) + 1) * config.time_interval

    current_uo = _interpolate_frame(uo, frame_times, times[0])
    current_vo = _interpolate_frame(vo, frame_times, times[0])

    particles = initialize_particles(
        config.batch_size,
        grid,
        current_uo,
        current_vo,
        random_generator,
    )

    latitudes = []
    longitudes = []
    inside_domains = []
    uo_at_particles = []
    vo_at_particles = []

    for i, current_time in enumerate(times):
        if i > 0:
            current_uo = _interpolate_frame(uo, frame_times, current_time)
            current_vo = _interpolate_frame(vo, frame_times, current_time)

        latitudes.append(particles.latitude.values.copy())
        longitudes.append(particles.longitude.values.copy())
        inside_domains.append(particles.inside_domain.values.copy())

        # Evolve one more step even for the last saved frame, so its uo/vo
        # are the actual interpolated velocities rather than a placeholder.
        particles, uo_particles, vo_particles = evolve_particles(
            particles,
            uo=current_uo * config.increased_speed_factor,
            vo=current_vo * config.increased_speed_factor,
            grid=grid,
            dt_seconds=dt_seconds,
        )
        uo_at_particles.append(uo_particles)
        vo_at_particles.append(vo_particles)

    # The velocities interpolated on the particles include the speed factor:
    # the saved ones are the physical ones, without it
    speed_factor = config.increased_speed_factor

    output = xr.Dataset(
        {
            "latitude": (("time", "particle"), np.stack(latitudes)),
            "longitude": (("time", "particle"), np.stack(longitudes)),
            "inside_domain": (("time", "particle"), np.stack(inside_domains)),
            "uo": (("time", "particle"), np.stack(uo_at_particles) / speed_factor),
            "vo": (("time", "particle"), np.stack(vo_at_particles) / speed_factor),
        },
        coords={
            "time": times,
            "particle": np.arange(config.batch_size),
        },
        attrs={"increased_speed_factor": speed_factor},
    )
    output_file = output_dir / f"particles_{batch_index}.nc"
    output.to_netcdf(output_file)
    return output_file


def compute_particles(
    domain: Domain,
    date: np.datetime64,
    input_dir: Path,
    output_dir: Path,
    *,
    config: ParticleConfig,
    show_progress: bool = True,
) -> None:
    """
    Compute the trajectories of all the batches of particles.

    Every batch is a dask task that receives only the frames of the velocity
    field it needs, read lazily from the zarr archive; this function does not
    choose how the tasks are executed: they run on the active dask scheduler
    (see `run_tasks`).

    Args:
        domain: The domain of the simulation.
        date: The date of the run of the model.
        input_dir: The directory with the data downloaded from the model.
        output_dir: Where the trajectories (and simulation.json) are written.
        config: The configuration of the simulation.
        show_progress: Whether to show a progress bar.
    """
    local_path = get_data_file_path(
        "velocities", input_dir, domain=domain, date=date, create_parents=False
    )
    ds = xr.open_zarr(local_path, chunks={})

    velocities = ds[["uo", "vo"]]

    source_times = ds.time.values
    grid = Grid(
        latitude=ds.latitude.values.astype(np.float64),
        longitude=ds.longitude.values.astype(np.float64),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    start_times = batch_start_times(source_times, config)
    seeds = np.random.SeedSequence(config.seed).spawn(len(start_times))
    batch_length = _n_steps(config) * config.time_interval

    tasks = []
    for batch_index, (start_time, seed) in enumerate(zip(start_times, seeds, strict=True)):
        window = _source_window(source_times, start_time, start_time + batch_length)
        window_velocities = velocities.isel(time=window)
        tasks.append(
            dask.delayed(run_batch)(
                window_velocities.uo.data,
                window_velocities.vo.data,
                source_times[window],
                grid,
                batch_index,
                start_time,
                output_dir,
                config,
                seed,
                dask_key_name=f"particles-batch-{batch_index}",
            )
        )

    run_tasks(tasks, show_progress=show_progress, description="Computing batches")

    current_simulation = LagrangianSimulation(
        domain=domain,
        date=numpy_to_datetime(date),
        config=config,
    )
    simulation_json = current_simulation.model_dump_json(indent=4)
    simulation_file = output_dir / "simulation.json"
    simulation_file.write_text(simulation_json, encoding="utf-8")
