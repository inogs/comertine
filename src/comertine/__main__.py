import argparse
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from comertine.time_resample import extend_ds_in_time, resample_ds_in_time
from comertine.domains import Domain
from comertine.domains import DOMAINS
from comertine.domains import Simulation

URL = "https://mer-model-bgc.ogs.it/output/${DOMAIN}/${DATE}-MER-MITgcmBFM-${DOMAIN}-fc-${FREQUENCY}-v01.zarr"
LOCAL_PATH = "/dev/shm/${DOMAIN}-${DATE}.zarr"

TITLE = "${DOMAIN} - Surface water currents - ${TIME}"

# The temporal distance from on frame to another
TIME_INTERVAL = np.timedelta64(120, "s")
TIME_INTERVAL_SECONDS = TIME_INTERVAL / np.timedelta64(1, "s")

# For how much time do we run a single particle?
BATCH_TIME_LENGTH = np.timedelta64(8, "h")

# The frequency when we span new particles
BATCH_EVERY = np.timedelta64(30, "m")

# How many particles to we spawn at the same time?
BATCH_SIZE = 500

# Dimensionless multipliers applied on top of the real TIME_INTERVAL to
# make particle motion visible on the animation (real advection over 120s
# would barely move the particles from one frame to the next).
VHR_SPEEDUP_FACTOR = 1.5
REGIONAL_SPEEDUP_FACTOR = 7.5
VISUAL_SPEEDUP_FACTOR = None

RANDOM_GENERATOR = np.random.default_rng(0)

PLOT_OUTPUT_DIR = Path("/dev/shm/particles_plot")

# Copernicus DEM GLO-30 tile covering the whole ocean model domain (a
# single 1x1 degree tile is enough, since the domain is tiny). Public,
# no-auth AWS Open Data bucket.
DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_N45_00_E013_00_DEM/"
    "Copernicus_DSM_COG_10_N45_00_E013_00_DEM.tif"
)
DEM_LOCAL_PATH = Path("/scratch/spiani/dem_n45_e013.tif")

PLOT_HILLSHADE_AZIMUTH = 315
PLOT_HILLSHADE_ALTITUDE = 45
PLOT_HILLSHADE_ZORDER = -1

# Plot resolution, sized for a fullscreen 4K display
PLOT_WIDTH_PX = 3840
PLOT_HEIGHT_PX = 2160
PLOT_DPI = 100

PLOT_TITLE_FONTSIZE = 28
PLOT_PARTICLE_SIZE = 15
PLOT_PARTICLE_COLOR = "white"
PLOT_PARTICLE_ALPHA = 0.8

REGIONAL_PLOT_GRIDLINE_STEP = 0.5
VHR_PLOT_GRIDLINE_STEP = 0.1
PLOT_GRIDLINE_STEP = None

PLOT_GRIDLINE_FONTSIZE = 20

# Colormap used to draw the current speed field.
# PLOT_SPEED_CMAP = "plasma"
PLOT_SPEED_CMAP = "viridis"

# Name of the cartopy.crs class used to draw the map axes (resolved lazily
# in plot_frame, where cartopy is imported). Data coordinates stay in plain
# lon/lat (ccrs.PlateCarree()) regardless of this setting.
# PLOT_PROJECTION = "PlateCarree"
# PLOT_PROJECTION = "Mercator"
PLOT_PROJECTION = "LambertConformal"
# PLOT_PROJECTION = "NearsidePerspective"

# Satellite altitude (in meters) used when PLOT_PROJECTION is
# "NearsidePerspective". Lower than the cartopy default (geostationary,
# ~35.8e6 m) so the globe's curvature actually reads on a small domain.
PLOT_NEARSIDE_SATELLITE_HEIGHT = 2_000_000

# Number of past positions (current one included) shown as a trail behind
# each particle. Truncated near the start of a particle's own batch file,
# where fewer past positions are known.
PLOT_TRAIL_LENGTH = 24
# Per-segment weight of the trail, ordered from the newest (closest to the
# current position) to the oldest. Must have PLOT_TRAIL_LENGTH - 1 entries.
# PLOT_TRAIL_SEGMENT_WEIGHTS = (1.0, 0.8, 0.66, 0.4, 0.33, 0.2, 0.1)
PLOT_TRAIL_SEGMENT_WEIGHTS = np.linspace(1, 0, PLOT_TRAIL_LENGTH - 1) ** 2
# How many frames after leaving the domain a particle's trail keeps fading
# out before disappearing entirely (its position is frozen once it exits,
# so no new segments are added during this window).
PLOT_TRAIL_FADE_OUT_FRAMES = 2
PLOT_TRAIL_LINEWIDTH = 2.0
PLOT_TRAIL_ZORDER = 1.5

# The dataset used by each worker process, set in main() before the
# multiprocessing pool is spawned so that fork-ed children inherit it.
_DS: xr.Dataset | None = None

# The particle datasets used by each plot worker process, set in
# plot_particles() before the multiprocessing pool is spawned so that
# fork-ed children inherit them.
_PARTICLE_DATASETS: list[xr.Dataset] | None = None

# The map extent (lon_min, lon_max, lat_min, lat_max) used by each plot
# worker process, set in plot_particles() before the multiprocessing pool is
# spawned so that fork-ed children inherit it.
_PLOT_EXTENT: tuple[float, float, float, float] | None = None

# The (resampled, non-extended) ocean dataset used by each plot worker
# process to draw the velocity field, set in plot_particles() before the
# multiprocessing pool is spawned so that fork-ed children inherit it.
_PLOT_DS: xr.Dataset | None = None

# The upper bound of the speed colorbar (0 is always the lower bound), set
# in plot_particles() before the multiprocessing pool is spawned so that
# fork-ed children inherit it.
_PLOT_SPEED_VMAX: float | None = 0.6

# The precomputed land image (either a grayscale hillshade or an
# already-colored hypsometric-tint hillshade) drawn under the ocean
# velocity field (whose NaN-over-land cells let it show through), as
# (image, lon, lat, cmap) - cmap is None when image is already RGBA. Set in
# plot_particles() before the multiprocessing pool is spawned so that
# fork-ed children inherit it; None when --land-filler is "none".
_PLOT_HILLSHADE: tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, str | None] | None = None

# Which coastline to draw: "none" (skip it), "GSHHFS" (full-resolution
# cfeature.GSHHSFeature), "cartopy10m"/"cartopy50m" (cartopy's own Natural
# Earth coastlines at 10m/50m resolution). Set in plot_particles() before
# the multiprocessing pool is spawned so that fork-ed children inherit it.
_PLOT_COASTLINE: str = "GSHHFS"


def get_borders_and_sizes(v: np.typing.NDArray):
    """
    Compute borders from the centers

    Given a 1D array that contains the position of the centers of some cells,
    it computes the position of the boundaries between the two cells supposing
    that the boundaries are exactly in the middle between the two centers.

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


def compute_new_coords(current_lat:np.ndarray, current_lon:np.ndarray, x_velocity:np.ndarray, y_velocity:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the new coordinates of the particles.

    Advects the current coordinates by the given velocity field over one
    simulation time step (TIME_INTERVAL_SECONDS), scaled by
    VISUAL_SPEEDUP_FACTOR so the motion is actually visible on the
    animation, then converts the resulting displacement from meters to
    degrees.

    Args:
        current_lat: The current latitude of the particle (in degrees).
        current_lon: The current longitude of the particle (in degrees).
        x_velocity: The velocity in the x direction (in m/s).
        y_velocity: The velocity in the y direction (in m/s).
    """
    # Earth radius in meters
    earth_radius = 6371000.0

    dt_seconds = TIME_INTERVAL_SECONDS * VISUAL_SPEEDUP_FACTOR
    y_movement = y_velocity * dt_seconds
    x_movement = x_velocity * dt_seconds

    # Convert y_movement (north-south) from meters to degrees
    lat_change = np.degrees(y_movement / earth_radius)

    # Convert x_movement (east-west) from meters to degrees
    # Account for convergence of meridians at different latitudes
    lon_change = np.degrees(
        x_movement / (earth_radius * np.cos(np.radians(current_lat))))

    new_lat = current_lat + lat_change
    new_lon = current_lon + lon_change

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

    particles["latitude_index"] = xr.DataArray(lat_index, dims=("particle",))
    particles["longitude_index"] = xr.DataArray(lon_index, dims=("particle",))


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

    prob_indices = np.sqrt(speed).ravel().cumsum()
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


def _bilinear_interpolation(
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
        return _bilinear_interpolation(
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


def _replace_string(template_string: str, domain: Domain, date: np.datetime64):
    replace_vals = {
        "${DOMAIN}": domain.name.upper(),
        "${DATE}": date.astype("datetime64[us]").item().strftime('%Y%m%d'),
        "${FREQUENCY}": "hourly"
    }
    for old, new in replace_vals.items():
        template_string = template_string.replace(old, new)
    return template_string

def _get_local_path(domain: Domain, date: np.datetime64):
    return Path(
        _replace_string(template_string=LOCAL_PATH, domain=domain, date=date)
    )

def _get_url(domain: Domain, date: np.datetime64):
    return _replace_string(template_string=URL, domain=domain, date=date)


def download_data(domain: Domain, date: np.datetime64):
    local_path = _get_local_path(domain, date)

    download_url = _get_url(domain, date)

    ds = xr.open_zarr(download_url, chunks={})

    ds = ds.squeeze("time", drop=True)
    ds = ds.swap_dims({"step": "valid_time"})
    ds = ds.drop_vars(["step"])
    ds = ds.rename({"valid_time": "time"})

    ds = ds.isel(depth=0)
    vars_to_keep = ["thetao", "uo", "vo", "so"]
    vars_to_drop = [d for d in ds.data_vars if d not in vars_to_keep]
    ds = ds.drop_vars(vars_to_drop)

    ds.to_zarr(local_path, mode="w")


def download_dem():
    if DEM_LOCAL_PATH.exists():
        return

    import requests

    DEM_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DEM_URL)
    response.raise_for_status()
    DEM_LOCAL_PATH.write_bytes(response.content)


def _load_dem_elevation(extent: tuple[float, float, float, float]):
    """
    Download (if needed) the DEM tile and clip it to the given
    (lon_min, lon_max, lat_min, lat_max) extent.

    Returns (elevation, lon, lat, dx, dy): the elevation grid, its
    coordinates, and its pixel spacing in metres (dx, dy) - accounting for
    the longitude convergence at this latitude - for use with
    matplotlib.colors.LightSource.
    """
    import rioxarray

    download_dem()

    lon_min, lon_max, lat_min, lat_max = extent
    dem = rioxarray.open_rasterio(DEM_LOCAL_PATH).squeeze("band", drop=True)
    dem = dem.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)

    elevation = dem.values.astype(float)
    lon = dem.x.values
    lat = dem.y.values

    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * np.cos(np.radians(lat.mean()))
    dx = abs(float(lon[1] - lon[0])) * meters_per_degree_lon
    dy = abs(float(lat[1] - lat[0])) * meters_per_degree_lat

    return elevation, lon, lat, dx, dy


def compute_hillshade(extent: tuple[float, float, float, float]):
    """
    Compute a grayscale hillshade of the land for the given
    (lon_min, lon_max, lat_min, lat_max) extent.

    Returns (hillshade, lon, lat, cmap) - hillshade has shape
    (len(lat), len(lon)) with values in [0, 1], to be drawn with cmap
    "gray".
    """
    from matplotlib.colors import LightSource

    elevation, lon, lat, dx, dy = _load_dem_elevation(extent)

    light_source = LightSource(
        azdeg=PLOT_HILLSHADE_AZIMUTH, altdeg=PLOT_HILLSHADE_ALTITUDE
    )
    hillshade = light_source.hillshade(elevation, dx=dx, dy=dy)

    return hillshade, lon, lat, "gray"


def compute_colored_hillshade(extent: tuple[float, float, float, float]):
    """
    Compute a hypsometric-tint hillshade of the land (elevation-based
    color, shaded by relief) for the given
    (lon_min, lon_max, lat_min, lat_max) extent.

    Returns (image, lon, lat, cmap) - image is an (len(lat), len(lon), 4)
    RGBA array already colored, so cmap is None (nothing left to map).
    """
    from matplotlib import colormaps
    from matplotlib.colors import LightSource, LinearSegmentedColormap

    elevation, lon, lat, dx, dy = _load_dem_elevation(extent)

    # matplotlib's "terrain" colormap goes ocean-blue -> green -> brown ->
    # white; since our DEM only covers land (elevation stays close to sea
    # level at worst), only its land portion is used, to avoid tinting low
    # ground blue as if it were water.
    terrain = colormaps["terrain"]
    land_cmap = LinearSegmentedColormap.from_list(
        "land_terrain", terrain(np.linspace(0.25, 1.0, 256))
    )

    light_source = LightSource(
        azdeg=PLOT_HILLSHADE_AZIMUTH, altdeg=PLOT_HILLSHADE_ALTITUDE
    )
    image = light_source.shade(
        elevation, cmap=land_cmap, blend_mode="soft",
        vmin=0, vert_exag=1, dx=dx, dy=dy,
    )

    return image, lon, lat, None



def run_batch(batch_index: int, batch_start_time: np.datetime64, output_dir: Path) -> None:
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

    if _DS is None:
        raise Exception("Dataset has not been initialized")

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
    output.to_netcdf(output_dir / f"particles_{batch_index}.nc")


def _run_batch_star(args: tuple[int, np.datetime64, Path]) -> None:
    run_batch(*args)


def compute_particles(domain: Domain, date: np.datetime64, output_dir: Path, show_progress: bool = True):
    global _DS
    global VISUAL_SPEEDUP_FACTOR

    if domain.resolution == "regional":
        VISUAL_SPEEDUP_FACTOR = REGIONAL_SPEEDUP_FACTOR
    elif domain.resolution == "vhr":
        VISUAL_SPEEDUP_FACTOR = VHR_SPEEDUP_FACTOR
    else:
        error_msg = f"Unknown resolution: {domain.resolution}"
        raise ValueError(error_msg)

    local_path = _get_local_path(domain, date)
    ds = xr.open_zarr(local_path, chunks={})

    ds = resample_ds_in_time(ds, TIME_INTERVAL)
    ds = extend_ds_in_time(ds, BATCH_TIME_LENGTH)
    # Loaded eagerly into memory here (once, in the parent process) so that
    # forked workers each get a plain numpy-backed dataset via copy-on-write,
    # instead of every worker independently re-computing the same dask graph.
    _DS = ds.compute()

    output_dir.mkdir(parents=True, exist_ok=True)

    time_values = _DS.time.values
    batch_start_times = []
    current_time = time_values[0]
    while current_time <= time_values[-1]:
        batch_start_times.append(current_time)
        current_time += BATCH_EVERY

    batches = []
    for batch_index, batch_start_time in enumerate(batch_start_times):
        batches.append((batch_index, batch_start_time, output_dir))

    with mp.get_context("fork").Pool() as pool, tqdm(
        total=len(batches), desc="Computing batches", disable=not show_progress
    ) as pbar:
        for _ in pool.imap_unordered(_run_batch_star, batches):
            pbar.update(1)

    current_simulation = Simulation(
        domain=domain,
        date=date.astype("datetime64[us]").item()
    )
    simulation_json = current_simulation.model_dump_json(indent=4)
    simulation_file = output_dir / "simulation.json"
    simulation_file.write_text(simulation_json, encoding="utf-8")


def plot_frame(index: int, current_time: np.datetime64, title: str) -> None:
    """
    Plot the particles (inside_domain only) at `current_time`, gathering
    them from every batch in `_PARTICLE_DATASETS` that covers that time,
    and save the result to PLOT_OUTPUT_DIR / f"plot{index}.png".

    `title` is the fully resolved plot title for this frame, computed by
    the caller.

    Runs in a worker process forked from plot_particles():
    `_PARTICLE_DATASETS` is inherited from the parent rather than passed as
    an argument.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    particle_rgb = to_rgba(PLOT_PARTICLE_COLOR)[:3]

    latitudes = []
    longitudes = []
    alphas = []
    trail_segments = []
    trail_colors = []

    if _PARTICLE_DATASETS is None:
        raise Exception("Datasets with particles have not been initialized")

    for particles in _PARTICLE_DATASETS:
        file_times = particles.time.values
        time_index = int(np.searchsorted(file_times, current_time))
        if time_index >= len(file_times) or file_times[time_index] != current_time:
            continue

        start_time = file_times[0]
        end_time = file_times[-1]

        def batch_fade_alpha(t):
            # Particles fade in/out over the first/last quarter of the time
            # span covered by their own batch file, so a whole batch doesn't
            # pop in/out abruptly.
            k = (t - start_time) / (end_time - start_time)
            if k < 0.25:
                return PLOT_PARTICLE_ALPHA * 4 * k
            if k > 0.75:
                return PLOT_PARTICLE_ALPHA * (1 - k) * 4
            return PLOT_PARTICLE_ALPHA

        frame = particles.isel(time=time_index)
        inside = frame.inside_domain.values
        n_particles = int(inside.sum())

        latitudes.append(frame.latitude.values[inside])
        longitudes.append(frame.longitude.values[inside])
        alphas.append(np.full(n_particles, batch_fade_alpha(current_time)))

        # --- trail ---
        window_start = max(0, time_index - PLOT_TRAIL_LENGTH + 1)
        window = particles.isel(time=slice(window_start, time_index + 1))
        window_lat = window.latitude.values
        window_lon = window.longitude.values
        window_inside = window.inside_domain.values
        n_window = window_lat.shape[0]
        if n_window < 2:
            continue

        # For each particle, how many frames back (from the current one) it
        # was last inside the domain: 0 means still active, 1 means it just
        # left, etc. Particles not found inside the domain anywhere in the
        # window left more than PLOT_TRAIL_LENGTH frames ago.
        reversed_inside = window_inside[::-1, :]
        has_any_true = reversed_inside.any(axis=0)
        frames_since_exit = np.argmax(reversed_inside, axis=0)

        draw_mask = has_any_true & (frames_since_exit <= PLOT_TRAIL_FADE_OUT_FRAMES)
        if not np.any(draw_mask):
            continue

        exit_fade = np.where(
            frames_since_exit == 0,
            1.0,
            np.clip(
                1.0 - frames_since_exit / (PLOT_TRAIL_FADE_OUT_FRAMES + 1), 0.0, 1.0
            ),
        )

        drawn = np.nonzero(draw_mask)[0]

        # (n_window, n_drawn, 2) -> (n_drawn, n_window, 2)
        points = np.stack(
            [window_lon[:, drawn], window_lat[:, drawn]], axis=-1
        ).transpose(1, 0, 2)
        segments = np.stack([points[:, :-1, :], points[:, 1:, :]], axis=2)

        n_segments = n_window - 1
        # Segment recency, counted from the current position backwards: 0 is
        # the newest segment (closest to "now"), which gets the highest
        # weight.
        segment_distance = n_segments - 1 - np.arange(n_segments)
        segment_weights = np.array(
            [PLOT_TRAIL_SEGMENT_WEIGHTS[d] for d in segment_distance]
        )
        window_times = file_times[window_start:time_index + 1]
        batch_alpha_per_step = np.array(
            [batch_fade_alpha(t) for t in window_times]
        )

        segment_alpha = (
            batch_alpha_per_step[1:][None, :]
            * segment_weights[None, :]
            * exit_fade[drawn][:, None]
        )

        trail_segments.append(segments.reshape(-1, 2, 2))
        segment_rgba = np.empty((segment_alpha.size, 4))
        segment_rgba[:, :3] = particle_rgb
        segment_rgba[:, 3] = segment_alpha.reshape(-1)
        trail_colors.append(segment_rgba)

    latitudes = np.concatenate(latitudes) if latitudes else np.empty(0)
    longitudes = np.concatenate(longitudes) if longitudes else np.empty(0)
    alphas = np.concatenate(alphas) if alphas else np.empty(0)

    ds_time_values = _PLOT_DS.time.values
    ds_time_index = int(np.searchsorted(ds_time_values, current_time))

    plot_ds = _PLOT_DS.copy(deep=False)
    uo = plot_ds.uo
    vo = plot_ds.vo
    plot_ds["speed"] = np.sqrt(uo * uo + vo * vo)

    lat_borders, _ = get_borders_and_sizes(_PLOT_DS.latitude.values)
    lon_borders, _ = get_borders_and_sizes(_PLOT_DS.longitude.values)

    plt.figure(
        figsize=(PLOT_WIDTH_PX / PLOT_DPI, PLOT_HEIGHT_PX / PLOT_DPI),
        dpi=PLOT_DPI,
    )
    if PLOT_PROJECTION == "LambertConformal":
        projection = ccrs.LambertConformal(
            central_longitude=(_PLOT_EXTENT[0] + _PLOT_EXTENT[1]) / 2,
            central_latitude=(_PLOT_EXTENT[2] + _PLOT_EXTENT[3]) / 2,
        )
    elif PLOT_PROJECTION == "NearsidePerspective":
        projection = ccrs.NearsidePerspective(
            central_longitude=(_PLOT_EXTENT[0] + _PLOT_EXTENT[1]) / 2,
            central_latitude=(_PLOT_EXTENT[2] + _PLOT_EXTENT[3]) / 2,
            satellite_height=PLOT_NEARSIDE_SATELLITE_HEIGHT,
        )
    else:
        projection = getattr(ccrs, PLOT_PROJECTION)()
    ax = plt.axes(projection=projection)
    ax.set_extent(_PLOT_EXTENT, crs=ccrs.PlateCarree())
    if _PLOT_HILLSHADE is not None:
        hillshade_image, hillshade_lon, hillshade_lat, hillshade_cmap = _PLOT_HILLSHADE
        origin = "upper" if hillshade_lat[0] > hillshade_lat[-1] else "lower"
        ax.imshow(
            hillshade_image, cmap=hillshade_cmap,
            vmin=0 if hillshade_cmap else None,
            vmax=1 if hillshade_cmap else None,
            origin=origin,
            extent=(
                hillshade_lon.min(), hillshade_lon.max(),
                hillshade_lat.min(), hillshade_lat.max(),
            ),
            transform=ccrs.PlateCarree(), zorder=PLOT_HILLSHADE_ZORDER,
        )

    data = plot_ds["speed"].isel(time=ds_time_index).values
    mesh = ax.pcolormesh(
        lon_borders, lat_borders, data,
        cmap=PLOT_SPEED_CMAP, vmin=0, vmax=_PLOT_SPEED_VMAX,
        transform=ccrs.PlateCarree(), zorder=0,
    )
    colorbar = plt.colorbar(mesh, ax=ax)
    colorbar.set_label("Speed [m/s]", fontsize=PLOT_TITLE_FONTSIZE)
    colorbar.ax.tick_params(labelsize=PLOT_TITLE_FONTSIZE)
    if _PLOT_COASTLINE == "GSHHFS":
        ax.add_feature(
            cfeature.GSHHSFeature(scale="full", edgecolor="black", facecolor="none"),
            zorder=1,
        )
    elif _PLOT_COASTLINE == "cartopy10m":
        ax.coastlines(resolution="10m", color="black", zorder=1)
    elif _PLOT_COASTLINE == "cartopy50m":
        ax.coastlines(resolution="50m", color="black", zorder=1)
    if trail_segments:
        trail_collection = LineCollection(
            np.concatenate(trail_segments, axis=0),
            colors=np.concatenate(trail_colors, axis=0),
            linewidths=PLOT_TRAIL_LINEWIDTH,
            transform=ccrs.PlateCarree(), zorder=PLOT_TRAIL_ZORDER,
        )
        ax.add_collection(trail_collection)
    ax.scatter(
        longitudes, latitudes, s=PLOT_PARTICLE_SIZE, marker='.', linewidths=0,
        color=PLOT_PARTICLE_COLOR, alpha=alphas,
        transform=ccrs.PlateCarree(), zorder=2,
    )

    gridlines = ax.gridlines(
        draw_labels=True, color="lightgray", linestyle="--", zorder=1,
        xlocs=mticker.MultipleLocator(PLOT_GRIDLINE_STEP),
        ylocs=mticker.MultipleLocator(PLOT_GRIDLINE_STEP),
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": PLOT_GRIDLINE_FONTSIZE}
    gridlines.ylabel_style = {"size": PLOT_GRIDLINE_FONTSIZE}
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    plt.savefig(PLOT_OUTPUT_DIR / f"plot{index:0>5}.png", dpi=PLOT_DPI)
    plt.close()


def _plot_frame_star(args: tuple[int, np.datetime64, str]) -> None:
    return plot_frame(*args)


def plot_particles(
        input_dir: Path,
        land_filler: str = "none",
        coastline: str = "GSHHFS",
        show_progress: bool = True,
):
    global _PARTICLE_DATASETS, _PLOT_EXTENT, _PLOT_DS, _PLOT_SPEED_VMAX, _PLOT_HILLSHADE, _PLOT_COASTLINE
    global PLOT_GRIDLINE_STEP

    _PLOT_COASTLINE = coastline

    simulation_file = input_dir / "simulation.json"
    simulation = Simulation.model_validate_json(simulation_file.read_text())

    domain = simulation.domain
    date = np.datetime64(simulation.date, 's')

    local_path = _get_local_path(domain=domain, date=date)

    ds = xr.open_zarr(local_path, chunks={})
    ds = resample_ds_in_time(ds, TIME_INTERVAL)
    # Kept lazy (dask-backed) and shared as-is with the forked workers: the
    # dataset can be hundreds of GB, so each worker reads only the time
    # slice it needs from disk instead of the whole thing being loaded into
    # memory here.
    _PLOT_DS = ds
    time_values = ds.time.values

    _PLOT_EXTENT = (
        float(ds.longitude.min()), float(ds.longitude.max()),
        float(ds.latitude.min()), float(ds.latitude.max()),
    )

    if domain.resolution == "regional":
        PLOT_GRIDLINE_STEP = REGIONAL_PLOT_GRIDLINE_STEP
    elif domain.resolution == "vhr":
        PLOT_GRIDLINE_STEP = VHR_PLOT_GRIDLINE_STEP
    else:
        error_msg = f"Unknown domain resolution: {domain.resolution}"
        raise ValueError(error_msg)

    if land_filler == "hillshade":
        _PLOT_HILLSHADE = compute_hillshade(_PLOT_EXTENT)
    elif land_filler == "colored-hillshade":
        _PLOT_HILLSHADE = compute_colored_hillshade(_PLOT_EXTENT)

    # Computed lazily through xarray/dask (a chunked reduction), so this
    # never materializes the full velocity field in memory at once.
    speed_max = np.sqrt(ds.uo ** 2 + ds.vo ** 2).max()
    if _PLOT_SPEED_VMAX is None:
        _PLOT_SPEED_VMAX = math.ceil(float(speed_max.compute()) * 10) / 10.

    # Loaded eagerly (and the files closed) here in the parent process, so
    # that forked workers each get plain in-memory datasets via
    # copy-on-write instead of sharing lazily-opened file handles.
    particle_datasets = []
    for f in input_dir.glob("particles_*.nc"):
        with xr.open_dataset(f) as particles:
            particle_datasets.append(particles.load())
    _PARTICLE_DATASETS = particle_datasets

    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    title_template = _replace_string(TITLE, domain=domain, date=date)
    frames = []
    for i, t in enumerate(time_values):
        rounded_time = (
            (t + np.timedelta64(30, "m"))
            .astype("datetime64[h]")
            .astype("datetime64[s]")
            .astype(object)
        )
        title = title_template.replace("${TIME}", rounded_time.strftime("%Y-%m-%d %H:%M"))
        frames.append((i, t, title))
    with mp.get_context("fork").Pool() as pool, tqdm(
        total=len(frames), desc="Plotting frames", disable=not show_progress
    ) as pbar:
        for _ in pool.imap_unordered(_plot_frame_star, frames):
            pbar.update(1)

def _parse_date_arg(value: str) -> np.datetime64:
    """Parse a "YYYY-MM-DD" CLI argument into an np.datetime64[s]."""
    return np.datetime64(value, "s")


def main():
    parser = argparse.ArgumentParser(prog="comertine")
    subparsers = parser.add_subparsers(dest="action", required=True)

    progress_parent = argparse.ArgumentParser(add_help=False)
    progress_parent.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="Show a progress bar (default: enabled).",
    )

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument(
        "-d", "--domain", choices=tuple(DOMAINS.keys()), required=True
    )
    download_parser.add_argument(
        "-t", "--date", type=_parse_date_arg, required=False, default=None,
        help="Date to compute, as YYYY-MM-DD (default: today).",
    )

    compute_parser = subparsers.add_parser("compute", parents=[progress_parent])
    compute_parser.add_argument(
        "-d", "--domain", choices=tuple(DOMAINS.keys()), required=True
    )
    compute_parser.add_argument(
        "-t", "--date", type=_parse_date_arg, required=False, default=None,
        help="Date to compute, as YYYY-MM-DD (default: today).",
    )
    compute_parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="The directory where the trajectories of the particles will be "
             "stored. If a flag like ${DOMAIN} or ${DATE} is inside the path, "
             "it will be replaced with the correct value",
    )

    plot_parser = subparsers.add_parser("plot", parents=[progress_parent])
    plot_parser.add_argument(
        "--land-filler", choices=["none", "hillshade", "colored-hillshade"],
        default="none",
    )
    plot_parser.add_argument(
        "--coastline", choices=["none", "GSHHFS", "cartopy10m", "cartopy50m"],
        default="GSHHFS",
    )
    plot_parser.add_argument(
        "-i", "--input-dir", type=Path, required=True,
        help="The directory where the trajectories of the particles are "
             "stored.",
    )

    args = parser.parse_args()

    if args.action == "download":
        domain = DOMAINS.get(args.domain)
        simulation_date = args.date
        if simulation_date is None:
            simulation_date = np.datetime64("today", "s")
        download_data(domain=domain, date=simulation_date)
    elif args.action == "compute":
        domain = DOMAINS.get(args.domain)
        simulation_date = args.date
        if simulation_date is None:
            simulation_date = np.datetime64("today", "s")
        output_dir = Path(
            _replace_string(
                args.output_dir,
                domain=domain,
                date=simulation_date
            )
        )
        compute_particles(
            domain=domain,
            date=simulation_date,
            output_dir=output_dir,
            show_progress=args.progress
        )
    elif args.action == "plot":
        plot_particles(
            input_dir=args.input_dir,
            land_filler=args.land_filler, coastline=args.coastline,
            show_progress=args.progress,
        )
    else:
        error_msg = f"Invalid action: {args.action}"
        raise ValueError(error_msg)


if __name__ == "__main__":
    main()
