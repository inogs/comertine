import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import xarray as xr

from comertine.time_resample import extend_ds_in_time, resample_ds_in_time


# URL = "https://mer-staging.exact-lab.it/output/SIC/20260321-MER-MITgcmBFM-SIC-fc-hourly-v01.zarr"
URL = "https://mer-staging.exact-lab.it/output/HGT/20260908-MER-MITgcmBFM-HGT-fc-hourly-v01.zarr/"

# The temporal distance from on frame to another
TIME_INTERVAL = np.timedelta64(150, "s")

# For how much time do we run a single particle
BATCH_TIME_LENGTH = np.timedelta64(8, "h")

# The frequency when we span new particles
BATCH_EVERY = np.timedelta64(2, "h")

# How many particles to we spawn at the same time
BATCH_SIZE = 2000

SPEED_FACTOR_INCREASE = 200


RANDOM_GENERATOR = np.random.default_rng(0)


LOCAL_PATH = Path("/scratch/spiani/hgt.zarr")
OUTPUT_DIR = Path("/dev/shm/particles/")
PLOT_OUTPUT_DIR = Path("/dev/shm/particles_plot")

# The dataset used by each worker process, set in main() before the
# multiprocessing pool is spawned so that fork-ed children inherit it.
_DS: xr.Dataset | None = None


def get_borders_and_sizes(v: np.typing.NDArray):
    """
    Compute borders from the centers

    Given a 1D array that contains the position of the centers of some cells,
    it computes the position of the boundaries between the two cells supposing
    that the boundary are exactly in the middle between the two centers.

    Args:
        v: the array with the position of the centers

    Returns:
        A tuple with two elements:
        - A new array with the position of the boundaries
        - A new array with the length of each cell (i.e., the distance between
          two consecutive boundaries)
    """
    v_borders = np.empty(v.shape[0] + 1, dtype=v.dtype)
    v_borders[1:-1] = (v[:-1] + v[1:]) / 2.
    v_borders[0] = v[0] - (v[1] - v[0]) / 2.
    v_borders[-1] = v[-1] + (v[-1] - v[-2]) / 2.
    return v_borders, v_borders[1:] - v_borders[:-1]


def compute_new_coords(current_lat:np.ndarray, current_lon:np.ndarray, x_movement:np.ndarray, y_movement:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the new coordinates of the particles.
    
    This function adds to the current coordinates the movement of the particles,
    taking care of converting the movements from meters to degrees.

    Args:
        current_lat: The current latitude of the particle (in degrees).
        current_lon: The current longitude of the particle (in degrees).
        x_movement: The movement in the x direction (in metres).
        y_movement: The movement in the y direction (in metres).
    """
    # Earth radius in meters
    EARTH_RADIUS = 6371000.0

    # Convert y_movement (north-south) from meters to degrees
    lat_change = np.degrees(y_movement / EARTH_RADIUS)

    # Convert x_movement (east-west) from meters to degrees
    # Account for convergence of meridians at different latitudes
    lon_change = np.degrees(
        x_movement / (EARTH_RADIUS * np.cos(np.radians(current_lat))))

    new_lat = current_lat + SPEED_FACTOR_INCREASE * lat_change
    new_lon = current_lon + SPEED_FACTOR_INCREASE * lon_change

    return new_lat, new_lon


def enlarge_array(v: np.typing.NDArray, fill_value: float= 0.):
    """
    Define a new array enlarged

    Creates a new array of the same data type and with the same elements as v,
    but each dimension is enlarged with a new element at the beginning and a new
    element at the end. The value of these new entries is `fill_value`.

    Args:
        v: the original array
        fill_value: the value to use to fill the new entries

    Return:
          A new array with 2 more elements for each dimension
    """
    new_shape = tuple(s+ 2 for s in v.shape)
    slices = tuple([slice(1, -1)] * len(v.shape))
    new_v = np.full(new_shape, fill_value, dtype=v.dtype)
    new_v[slices] = v
    return new_v


def get_particles_cells(particles: xr.Dataset, latitude, longitude):
    """
    Compute the indices where the particles are

    Args:
        particles: a Dataset with the position of the particles
        latitude: the latitude of the centers of the cells
        longitude: the longitude of the centers of the cells

    """
    particle_lat = particles["latitude"].values
    particle_lon = particles["longitude"].values

    latitude_borders, _ = get_borders_and_sizes(latitude)
    longitude_borders, _ = get_borders_and_sizes(longitude)

    lat_index = np.searchsorted(latitude_borders[:-1], particle_lat) - 1
    lon_index = np.searchsorted(longitude_borders[:-1], particle_lon) - 1

    particles["latitude_index"] = xr.DataArray(lat_index, dims=("particles",))
    particles["longitude_index"] = xr.DataArray(lon_index, dims=("particles",))


def initialize_particles(n_particles, latitude, longitude, uo, vo,):
    latitude_borders, lat_sizes = get_borders_and_sizes(latitude)
    longitude_borders, lon_sizes = get_borders_and_sizes(longitude)

    data_mask = np.isfinite(uo)

    cells_lat_indices = np.arange(0, latitude.shape[0])[:, None]
    cells_lon_indices = np.arange(0, longitude.shape[0])[None, :]

    cells_lat_indices, cells_lon_indices = np.broadcast_arrays(cells_lat_indices, cells_lon_indices)

    cells_lat_indices = cells_lat_indices[data_mask]
    cells_lon_indices = cells_lon_indices[data_mask]

    speed = np.sqrt(uo[data_mask]**2 + vo[data_mask]**2)
    speed_indices = speed.ravel().cumsum()

    prob_indices = np.sqrt(speed_indices + 1) - 1
    samples = RANDOM_GENERATOR.uniform(0, prob_indices[-1], size=n_particles)

    shift_lat = RANDOM_GENERATOR.uniform(0, 1, size=n_particles)
    shift_lon = RANDOM_GENERATOR.uniform(0, 1, size=n_particles)

    data_cells = np.searchsorted(prob_indices, samples)

    samples_lat_cells = cells_lat_indices[data_cells]
    samples_lon_cells = cells_lon_indices[data_cells]

    lat_position = latitude_borders[samples_lat_cells]
    lon_position = longitude_borders[samples_lon_cells]

    lat_position += lat_sizes[samples_lat_cells] * shift_lat
    lon_position += lon_sizes[samples_lon_cells] * shift_lon

    ds = xr.Dataset(
        {
            "latitude": xr.DataArray(lat_position, dims=("particle",)),
            "longitude": xr.DataArray(lon_position, dims=("particle",)),
            "inside_domain": xr.DataArray(np.ones(n_particles, dtype=bool), dims=("particle",)),
        }
    )

    get_particles_cells(ds, latitude, longitude)
    return ds


def _linearly_interpolate_between(p1, p2, q1, q2, t):
    return (q1 * (p2 - t) + q2 * (t - p1)) / (p2 - p1)


def _bilinear_inteprolation(
        lat_min, lat_max, lon_min, lon_max, v_min_min, v_max_min,
        v_min_max, v_max_max, lat_position, lon_position):
    q1 = _linearly_interpolate_between(lat_min, lat_max, v_min_min, v_max_min, lat_position)
    q2 = _linearly_interpolate_between(lat_min, lat_max, v_min_max, v_max_max, lat_position)
    return _linearly_interpolate_between(lon_min, lon_max, q1, q2, lon_position)


def evolve_particles(particles, uo, vo, lat_grid, lon_grid):
    uo_enlarged = enlarge_array(np.where(np.isfinite(uo), uo, 0))
    vo_enlarged = enlarge_array(np.where(np.isfinite(vo), vo, 0))

    latitude_borders, _ = get_borders_and_sizes(lat_grid)
    longitude_borders, _ = get_borders_and_sizes(lon_grid)

    grid_lat_min = np.min(latitude_borders)
    grid_lat_max = np.max(latitude_borders)

    grid_lon_min = np.min(longitude_borders)
    grid_lon_max = np.max(longitude_borders)

    lat_enlarged = enlarge_array(lat_grid, fill_value=grid_lat_min - 1)
    lon_enlarged = enlarge_array(lon_grid, fill_value=grid_lon_min - 1)
    lat_enlarged[-1] = grid_lat_max + 1
    lon_enlarged[-1] = grid_lon_max + 1

    latitude_borders, _ = get_borders_and_sizes(lat_grid)
    longitude_borders, _ = get_borders_and_sizes(lon_grid)

    latitude = particles.latitude.values
    longitude = particles.longitude.values
    inside_domain = particles.inside_domain.values
    latitude_index = particles.latitude_index.values
    longitude_index = particles.longitude_index.values

    # Particles that have already left the domain (grid bounds or land)
    # stay where they are: only the active ones are evolved.
    active = inside_domain

    active_latitude = latitude[active]
    active_longitude = longitude[active]
    active_latitude_index = latitude_index[active]
    active_longitude_index = longitude_index[active]

    reference_center_lat = lat_grid[active_latitude_index]
    reference_center_lon = lon_grid[active_longitude_index]

    lat_min_index = np.where(
        active_latitude < reference_center_lat,
        active_latitude_index,
        active_latitude_index + 1
    )
    lon_min_index = np.where(
        active_longitude < reference_center_lon,
        active_longitude_index,
        active_longitude_index + 1
    )

    lat_min = lat_enlarged[lat_min_index]
    lon_min = lon_enlarged[lon_min_index]
    lat_max = lat_enlarged[lat_min_index + 1]
    lon_max = lon_enlarged[lon_min_index + 1]

    assert np.all(active_latitude >= lat_min)
    assert np.all(active_latitude <= lat_max)
    assert np.all(active_longitude >= lon_min)
    assert np.all(active_longitude <= lon_max)

    def _interpolate_variable_on_particles(var_data):
        return _bilinear_inteprolation(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            v_min_min=var_data[lat_min_index, lon_min_index],
            v_max_min=var_data[lat_min_index + 1, lon_min_index],
            v_min_max=var_data[lat_min_index, lon_min_index + 1],
            v_max_max=var_data[lat_min_index + 1, lon_min_index + 1],
            lat_position=active_latitude,
            lon_position=active_longitude,
        )

    uo_particles =  _interpolate_variable_on_particles(uo_enlarged)
    vo_particles = _interpolate_variable_on_particles(vo_enlarged)

    particles_speed = np.sqrt(
        uo_particles * uo_particles + vo_particles * vo_particles
    )

    new_active_lat, new_active_lon = compute_new_coords(
        active_latitude,
        active_longitude,
        uo_particles,
        vo_particles,
    )

    inside_lat = np.logical_and(
        new_active_lat >= grid_lat_min,
        new_active_lat <= grid_lat_max,
    )
    inside_lon = np.logical_and(
        new_active_lon >= grid_lon_min,
        new_active_lon <= grid_lon_max,
    )
    still_inside = np.logical_and(inside_lat, inside_lon)

    # A particle that landed on a cell that was originally NaN (land) has
    # left the domain too, even though it is still within the grid bounds.
    new_active_lat_index = np.searchsorted(latitude_borders[:-1], new_active_lat) - 1
    new_active_lon_index = np.searchsorted(longitude_borders[:-1], new_active_lon) - 1
    on_land = ~np.isfinite(uo[new_active_lat_index, new_active_lon_index])
    still_inside = np.logical_and(still_inside, ~on_land)

    new_latitude = latitude.copy()
    new_longitude = longitude.copy()
    new_inside_domain = inside_domain.copy()

    new_latitude[active] = new_active_lat
    new_longitude[active] = new_active_lon
    new_inside_domain[active] = still_inside

    uo_at_particles = np.full(latitude.shape, np.nan)
    vo_at_particles = np.full(latitude.shape, np.nan)
    uo_at_particles[active] = uo_particles
    vo_at_particles[active] = vo_particles

    new_particles = xr.Dataset(
        {
            "latitude": xr.DataArray(new_latitude, dims=("particle",)),
            "longitude": xr.DataArray(new_longitude, dims=("particle",)),
            "inside_domain": xr.DataArray(new_inside_domain, dims=("particle",)),
        }
    )
    get_particles_cells(new_particles, lat_grid, lon_grid)
    return new_particles, uo_at_particles, vo_at_particles


def download_data():
    if LOCAL_PATH.exists():
        return

    ds = xr.open_zarr(URL, chunks={})

    ds = ds.squeeze("time", drop=True)
    ds = ds.swap_dims({"step": "valid_time"})
    ds = ds.drop_vars(["step"])
    ds = ds.rename({"valid_time": "time"})

    ds = ds.isel(depth=0)
    vars_to_keep = ["thetao", "uo", "vo"]
    vars_to_drop = [d for d in ds.data_vars if d not in vars_to_keep]
    ds = ds.drop_vars(vars_to_drop)
    print(ds)

    ds.to_zarr(LOCAL_PATH, mode="w")


def main__ld():
    download_data()

    ds = xr.open_zarr(LOCAL_PATH, chunks={})

    ds = resample_ds_in_time(ds, TIME_INTERVAL)

    from matplotlib import pyplot as plt
    for i in range(100):
        positions = initialize_particles(
            BATCH_SIZE,
            ds.latitude.values,
            ds.longitude.values,
            ds.uo.isel(time=0).values,
            ds.vo.isel(time=0).values
        )
        lats = positions["latitude"]
        lons = positions["longitude"]
        plt.scatter(lons, lats, s=1, marker='.', linewidths=0)
        plt.savefig(f"/dev/shm/particles_{i:0>4}.png")
        plt.close()


def run_batch(batch_index: int, batch_start_time: np.datetime64) -> None:
    """
    Spawn a batch of particles at `batch_start_time` and evolve them for
    BATCH_TIME_LENGTH, saving the result to
    OUTPUT_DIR / f"particles_{batch_index}.nc".

    Runs in a worker process forked from main(): `_DS` is inherited from
    the parent rather than passed as an argument, and RANDOM_GENERATOR is
    reseeded here so that parallel batches don't replay the same random
    sequence inherited from the fork.
    """
    global RANDOM_GENERATOR
    RANDOM_GENERATOR = np.random.default_rng(batch_index)

    ds = _DS
    time_values = ds.time.values
    n_time = ds.sizes["time"]

    start_idx = int(np.searchsorted(time_values, batch_start_time))

    particles = initialize_particles(
        BATCH_SIZE,
        ds.latitude.values,
        ds.longitude.values,
        ds.uo.isel(time=start_idx).values,
        ds.vo.isel(time=start_idx).values,
    )

    times = []
    latitudes = []
    longitudes = []
    inside_domains = []
    uo_at_particles = []
    vo_at_particles = []

    i = 0
    while True:
        current_time = batch_start_time + i * TIME_INTERVAL
        elapsed = current_time - batch_start_time

        times.append(current_time)
        latitudes.append(particles.latitude.values.copy())
        longitudes.append(particles.longitude.values.copy())
        inside_domains.append(particles.inside_domain.values.copy())

        # Evolve one more step even for the last saved frame, so its uo/vo
        # are the actual interpolated velocities rather than a placeholder.
        time_index = min(start_idx + i, n_time - 1)
        particles, uo_particles, vo_particles = evolve_particles(
            particles,
            uo=ds.uo.isel(time=time_index).values,
            vo=ds.vo.isel(time=time_index).values,
            lat_grid=ds.latitude.values,
            lon_grid=ds.longitude.values,
        )
        uo_at_particles.append(uo_particles)
        vo_at_particles.append(vo_particles)

        if elapsed >= BATCH_TIME_LENGTH:
            break

        i += 1

    output = xr.Dataset(
        {
            "latitude": (("time", "particle"), np.stack(latitudes)),
            "longitude": (("time", "particle"), np.stack(longitudes)),
            "inside_domain": (("time", "particle"), np.stack(inside_domains)),
            "uo": (("time", "particle"), np.stack(uo_at_particles)),
            "vo": (("time", "particle"), np.stack(vo_at_particles)),
        },
        coords={
            "time": np.array(times),
            "particle": np.arange(BATCH_SIZE),
        },
    )
    output.to_netcdf(OUTPUT_DIR / f"particles_{batch_index}.nc")


def compute_particles():
    global _DS

    download_data()

    ds = xr.open_zarr(LOCAL_PATH, chunks={})

    ds = resample_ds_in_time(ds, TIME_INTERVAL)
    ds = extend_ds_in_time(ds, BATCH_TIME_LENGTH)
    # Loaded eagerly into memory here (once, in the parent process) so that
    # forked workers each get a plain numpy-backed dataset via copy-on-write,
    # instead of every worker independently re-computing the same dask graph.
    _DS = ds.compute()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_values = _DS.time.values
    batch_start_times = []
    current_time = time_values[0]
    while current_time <= time_values[-1]:
        batch_start_times.append(current_time)
        current_time += BATCH_EVERY

    with mp.get_context("fork").Pool() as pool:
        pool.starmap(run_batch, enumerate(batch_start_times))


def plot_particles():
    from matplotlib import pyplot as plt

    download_data()

    ds = xr.open_zarr(LOCAL_PATH, chunks={})
    ds = resample_ds_in_time(ds, TIME_INTERVAL)
    time_values = ds.time.values

    particle_datasets = [
        xr.open_dataset(f) for f in OUTPUT_DIR.glob("particles_*.nc")
    ]

    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, current_time in enumerate(time_values):
        latitudes = []
        longitudes = []

        for particles in particle_datasets:
            file_times = particles.time.values
            time_index = int(np.searchsorted(file_times, current_time))
            if time_index >= len(file_times) or file_times[time_index] != current_time:
                continue

            frame = particles.isel(time=time_index)
            inside = frame.inside_domain.values
            latitudes.append(frame.latitude.values[inside])
            longitudes.append(frame.longitude.values[inside])

        latitudes = np.concatenate(latitudes) if latitudes else np.empty(0)
        longitudes = np.concatenate(longitudes) if longitudes else np.empty(0)

        plt.scatter(longitudes, latitudes, s=1, marker='.', linewidths=0)
        plt.savefig(PLOT_OUTPUT_DIR / f"plot{i}.png")
        plt.close()

    for particles in particle_datasets:
        particles.close()


def main():
    parser = argparse.ArgumentParser(prog="comertine")
    parser.add_argument("action", choices=["compute", "plot"])
    args = parser.parse_args()

    if args.action == "compute":
        compute_particles()
    else:
        plot_particles()


if __name__ == "__main__":
    main()
