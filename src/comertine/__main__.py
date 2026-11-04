import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from distributed import Client
from distributed import LocalCluster
from tqdm import tqdm

from comertine.algorithms.lagrangian_simulation import LagrangianSimulation
from comertine.algorithms.lagrangian_simulation import compute_particles
from comertine.tools.file_names import concretize_template_string
from comertine.tools.file_names import get_data_file_path
from comertine.tools.geometry import Extent
from comertine.tools.geometry import get_borders_and_sizes
from comertine.tools.time_resample import resample_ds_in_time
from comertine.domains import DOMAINS
from comertine.domains import Domain
from comertine.tools.cli import read_cli_args
from comertine.tools.downloader import download_data
from comertine.tools.particle_config import ParticleConfig
from comertine.tools.particle_config import auto_speed_factor
from comertine.tools.plots import make_projection
from comertine.tools.plots import resolve_colormap
from comertine.tools.static_data import check_static_data
from comertine.tools.static_data import configure_cartopy
from comertine.tools.static_data import download_static_data
from comertine.tools.static_data import get_dem_product
from comertine.tools.static_data import get_dem_tile_paths


TITLE = "${DOMAIN} - ${VARNAME} - ${TIME}"


RANDOM_GENERATOR = np.random.default_rng(0)

PLOT_HILLSHADE_AZIMUTH = 315
PLOT_HILLSHADE_ALTITUDE = 45
PLOT_HILLSHADE_ZORDER = -1

# Plot resolution, sized for a fullscreen 4K display
PLOT_WIDTH_PX = 3840
PLOT_HEIGHT_PX = 2160
PLOT_DPI = 100

# Margins (in pixels) around the map used when --tight-layout is enabled:
# the figure keeps PLOT_HEIGHT_PX as its height, the map fills the space
# left by the top/bottom margins and the figure width is recomputed from
# the map's aspect ratio, plus room for the latitude labels on the left and
# for the colorbar (with its tick labels and label) on the right.
PLOT_TIGHT_MARGIN_LEFT_PX = 130
PLOT_TIGHT_MARGIN_RIGHT_PX = 220
PLOT_TIGHT_MARGIN_TOP_PX = 90
PLOT_TIGHT_MARGIN_BOTTOM_PX = 70
PLOT_TIGHT_COLORBAR_PAD_PX = 50
PLOT_TIGHT_COLORBAR_WIDTH_PX = 60
# The recomputed figure width is rounded up to a multiple of this value.
PLOT_TIGHT_WIDTH_MULTIPLE = 8

PLOT_TITLE_FONTSIZE = 28
PLOT_PARTICLE_SIZE = 15
PLOT_PARTICLE_COLOR = "white"
PLOT_PARTICLE_ALPHA = 0.8

# Color of the coastline, in the default style and with --dark-mode.
PLOT_COASTLINE_COLOR = "black"
PLOT_COASTLINE_COLOR_DARK = "white"

# Logo drawn inside the map axes, in the bottom right corner. The path is
# relative to the current working directory; if the file does not exist,
# no logo is drawn. The logo is scaled (keeping its aspect ratio) to fit a
# box PLOT_LOGO_SIZE times the size of the map axes, placed
# PLOT_LOGO_PADDING (in axes coordinates) away from the bottom and right
# borders, so it never sticks out of the axes.
PLOT_LOGO_PATH = Path("../logo.png")
PLOT_LOGO_SIZE = 1 / 6
PLOT_LOGO_PADDING = 0.01
PLOT_LOGO_ZORDER = 3

REGIONAL_PLOT_GRIDLINE_STEP = 0.5
VHR_PLOT_GRIDLINE_STEP = 0.1
PLOT_GRIDLINE_STEP = None

PLOT_GRIDLINE_FONTSIZE = 20

# Colorbar label for each supported --background-variable choice.
PLOT_BACKGROUND_LABELS = {
    "speed": "Speed [m/s]",
    "thetao": "Temperature [°C]",
    "so": "Salinity [PSU]",
}

# Name of the cartopy.crs class used to draw the map axes (see
# comertine.tools.plots.make_projection). Data coordinates stay in plain
# lon/lat (ccrs.PlateCarree()) regardless of this setting.
# PLOT_PROJECTION = "PlateCarree"
PLOT_PROJECTION = "Mercator"
# PLOT_PROJECTION = "LambertConformal"
# PLOT_PROJECTION = "NearsidePerspective"

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

# The variable drawn as the background color field: "speed" (computed from
# uo/vo) or a raw dataset variable such as "thetao"/"so". Set in
# plot_particles() before the multiprocessing pool is spawned so that
# fork-ed children inherit it.
_PLOT_BACKGROUND_VARIABLE: str | None = None

# The lower/upper bounds of the background field's colorbar, set in
# plot_particles() before the multiprocessing pool is spawned so that
# fork-ed children inherit them. When not given on the command line,
# computed from the dataset as the 5th/95th percentile of the background
# field's values, respectively.
_PLOT_SPEED_VMIN: float | None = None
_PLOT_SPEED_VMAX: float | None = None

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

# Name of the colormap used to draw the background field, as a plain
# matplotlib colormap name (e.g. "viridis") or, prefixed with "cmo:", the
# name of a cmocean colormap (e.g. "cmo:thermal"). Turned into an actual
# Colormap object by resolve_colormap() in each plot worker. Set in
# plot_particles() before the multiprocessing pool is spawned so that
# fork-ed children inherit it.
PLOT_BACKGROUND_CMAP: str = "viridis"

# Figure geometry used when --tight-layout is enabled, as
# (figure_width_px, map_axes_rect, colorbar_axes_rect), where the rects are
# (left, bottom, width, height) in figure coordinates; None otherwise (the
# default PLOT_WIDTH_PX x PLOT_HEIGHT_PX canvas is used). Set in
# plot_particles() before the multiprocessing pool is spawned so that
# fork-ed children inherit it.
_PLOT_TIGHT_LAYOUT: tuple[int, tuple[float, float, float, float], tuple[float, float, float, float]] | None = None

# Whether to draw the frames with matplotlib's "dark_background" style. Set
# in plot_particles() before the multiprocessing pool is spawned so that
# fork-ed children inherit it.
_PLOT_DARK_MODE: bool = False

# The logo image (as returned by matplotlib.image.imread) drawn in the
# bottom right corner of the map axes, or None if PLOT_LOGO_PATH does not exist.
# Loaded in plot_particles() before the multiprocessing pool is spawned so
# that fork-ed children inherit it.
_PLOT_LOGO: np.typing.NDArray | None = None



def _load_dem_elevation(extent: tuple[float, float, float, float], tile_paths: list[Path]):
    """
    Read the DEM tiles stored in tile_paths, clip them to the given
    (lon_min, lon_max, lat_min, lat_max) extent and merge them into a
    single grid. The areas not covered by any tile (the tiles that would
    only contain sea do not exist) are filled with 0, i.e. sea level.

    Returns (elevation, lon, lat, dx, dy): the elevation grid, its
    coordinates, and its pixel spacing in metres (dx, dy) - accounting for
    the longitude convergence at this latitude - for use with
    matplotlib.colors.LightSource - or None if no tile intersects the
    extent.
    """
    import rioxarray
    from rioxarray.exceptions import NoDataInBounds
    from rioxarray.merge import merge_arrays

    lon_min, lon_max, lat_min, lat_max = extent
    pieces = []
    for tile_path in tile_paths:
        tile = rioxarray.open_rasterio(tile_path).squeeze("band", drop=True)
        try:
            pieces.append(
                tile.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)
            )
        except NoDataInBounds:
            continue
    if not pieces:
        return None

    # Downsample the DEM (averaging the pixels) so that it is not larger
    # than the frame: the map always fits inside PLOT_WIDTH_PX x
    # PLOT_HEIGHT_PX keeping its aspect ratio, so scaling the DEM by
    # min(W / nx, H / ny) still leaves at least one DEM pixel for each
    # pixel of the map. Without this, the hillshade of a regional domain at
    # 90 m is wider than the whole 4K frame and needs several GB of memory.
    # Each tile is downsampled on its own, before merging them, so that the
    # full resolution grid of the whole domain is never built.
    res_x, res_y = (abs(r) for r in pieces[0].rio.resolution())
    full_nx = (lon_max - lon_min) / res_x
    full_ny = (lat_max - lat_min) / res_y
    scale = min(PLOT_WIDTH_PX / full_nx, PLOT_HEIGHT_PX / full_ny)
    if scale < 1:
        from rasterio.enums import Resampling

        res_x, res_y = res_x / scale, res_y / scale
        pieces = [
            # No nodata value: the sea (0) must be averaged with the land,
            # not skipped
            piece.rio.write_nodata(None).rio.reproject(
                piece.rio.crs, resolution=(res_x, res_y),
                resampling=Resampling.average,
            )
            for piece in pieces
        ]

    # Merged on a single grid covering the whole extent: the areas not
    # covered by any tile are filled with 0 (sea level)
    dem = merge_arrays(
        pieces, bounds=(lon_min, lat_min, lon_max, lat_max),
        res=(res_x, res_y), nodata=0,
    )

    elevation = dem.values.astype(float)
    lon = dem.x.values
    lat = dem.y.values

    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * np.cos(np.radians(lat.mean()))
    dx = abs(float(lon[1] - lon[0])) * meters_per_degree_lon
    dy = abs(float(lat[1] - lat[0])) * meters_per_degree_lat

    return elevation, lon, lat, dx, dy


def compute_hillshade(extent: tuple[float, float, float, float], tile_paths: list[Path]):
    """
    Compute a grayscale hillshade of the land for the given
    (lon_min, lon_max, lat_min, lat_max) extent.

    Returns (hillshade, lon, lat, cmap) - hillshade has shape
    (len(lat), len(lon)) with values in [0, 1], to be drawn with cmap
    "gray".
    """
    from matplotlib.colors import LightSource

    dem = _load_dem_elevation(extent, tile_paths)
    if dem is None:
        return None
    elevation, lon, lat, dx, dy = dem

    light_source = LightSource(
        azdeg=PLOT_HILLSHADE_AZIMUTH, altdeg=PLOT_HILLSHADE_ALTITUDE
    )
    hillshade = light_source.hillshade(elevation, dx=dx, dy=dy)

    return hillshade, lon, lat, "gray"


def compute_colored_hillshade(extent: tuple[float, float, float, float], tile_paths: list[Path]):
    """
    Compute a hypsometric-tint hillshade of the land (elevation-based
    color, shaded by relief) for the given
    (lon_min, lon_max, lat_min, lat_max) extent.

    Returns (image, lon, lat, cmap) - image is an (len(lat), len(lon), 4)
    RGBA array already colored, so cmap is None (nothing left to map).
    """
    from matplotlib import colormaps
    from matplotlib.colors import LightSource, LinearSegmentedColormap

    dem = _load_dem_elevation(extent, tile_paths)
    if dem is None:
        return None
    elevation, lon, lat, dx, dy = dem

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



def _select_background_field(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """
    Select the DataArray to draw as the background color field.

    Returns the speed magnitude computed from uo/vo when `variable` is
    "speed", or the raw dataset variable otherwise (e.g. "thetao", "so").
    """
    if variable == "speed":
        return np.sqrt(ds.uo ** 2 + ds.vo ** 2)
    return ds[variable]


def _get_background_variable_name(ds: xr.Dataset, variable: str) -> str:
    """
    Resolve the ${VARNAME} placeholder in TITLE for the given background
    variable.

    Fixed to "Sea water currents" for "speed" (which is computed from
    uo/vo and so has no long_name of its own); otherwise the raw dataset
    variable's "long_name" attribute. Raises a ValueError if that variable
    has no "long_name" attribute.
    """
    if variable == "speed":
        return "Sea water currents"

    long_name = ds[variable].attrs.get("long_name")
    if long_name is None:
        error_msg = f"Variable '{variable}' has no 'long_name' attribute"
        raise ValueError(error_msg)
    return long_name




def _compute_tight_layout(
        extent: tuple[float, float, float, float]
) -> tuple[int, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """
    Compute the figure geometry used by --tight-layout: the figure keeps
    PLOT_HEIGHT_PX as its height and its width is derived from the aspect
    ratio of the map (in projected coordinates) once `extent` is set, plus
    the PLOT_TIGHT_* margins.

    Returns (figure_width_px, map_axes_rect, colorbar_axes_rect), with the
    rects as (left, bottom, width, height) in figure coordinates.
    """
    import cartopy.crs as ccrs
    # A bare Figure (not pyplot) so that no GUI backend gets initialized in
    # the parent process before the workers are forked.
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_axes((0, 0, 1, 1), projection=make_projection(PLOT_PROJECTION, Extent(*extent)))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    map_height_px = PLOT_HEIGHT_PX - PLOT_TIGHT_MARGIN_TOP_PX - PLOT_TIGHT_MARGIN_BOTTOM_PX
    map_width_px = map_height_px * aspect_ratio
    figure_width_px = (
        PLOT_TIGHT_MARGIN_LEFT_PX + map_width_px + PLOT_TIGHT_COLORBAR_PAD_PX
        + PLOT_TIGHT_COLORBAR_WIDTH_PX + PLOT_TIGHT_MARGIN_RIGHT_PX
    )
    # Rounded up to a multiple of PLOT_TIGHT_WIDTH_MULTIPLE pixels, so that
    # the frames are friendly to video encoders (chroma subsampling and
    # macroblocks) when turning them into a movie.
    figure_width_px = int(
        np.ceil(figure_width_px / PLOT_TIGHT_WIDTH_MULTIPLE) * PLOT_TIGHT_WIDTH_MULTIPLE
    )

    map_rect = (
        PLOT_TIGHT_MARGIN_LEFT_PX / figure_width_px,
        PLOT_TIGHT_MARGIN_BOTTOM_PX / PLOT_HEIGHT_PX,
        map_width_px / figure_width_px,
        map_height_px / PLOT_HEIGHT_PX,
    )
    colorbar_rect = (
        (PLOT_TIGHT_MARGIN_LEFT_PX + map_width_px + PLOT_TIGHT_COLORBAR_PAD_PX) / figure_width_px,
        PLOT_TIGHT_MARGIN_BOTTOM_PX / PLOT_HEIGHT_PX,
        PLOT_TIGHT_COLORBAR_WIDTH_PX / figure_width_px,
        map_height_px / PLOT_HEIGHT_PX,
    )
    return figure_width_px, map_rect, colorbar_rect


def plot_frame(index: int, current_time: np.datetime64, title: str, output_dir: Path) -> None:
    """
    Plot the particles (inside_domain only) at `current_time`, gathering
    them from every batch in `_PARTICLE_DATASETS` that covers that time,
    and save the result to output_dir / f"plot{index}.png".

    `title` is the fully resolved plot title for this frame, and
    `output_dir` the directory to save it into; both computed by the
    caller.

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

    background_field = _select_background_field(_PLOT_DS, _PLOT_BACKGROUND_VARIABLE)

    lat_borders, _ = get_borders_and_sizes(_PLOT_DS.latitude.values)
    lon_borders, _ = get_borders_and_sizes(_PLOT_DS.longitude.values)

    if _PLOT_DARK_MODE:
        plt.style.use("dark_background")

    if _PLOT_TIGHT_LAYOUT is not None:
        figure_width_px, map_rect, colorbar_rect = _PLOT_TIGHT_LAYOUT
    else:
        figure_width_px, map_rect, colorbar_rect = PLOT_WIDTH_PX, None, None
    fig = plt.figure(
        figsize=(figure_width_px / PLOT_DPI, PLOT_HEIGHT_PX / PLOT_DPI),
        dpi=PLOT_DPI,
    )
    projection = make_projection(PLOT_PROJECTION, Extent(*_PLOT_EXTENT))
    if map_rect is not None:
        ax = fig.add_axes(map_rect, projection=projection)
    else:
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

    data = background_field.isel(time=ds_time_index).values
    background_cmap = resolve_colormap(PLOT_BACKGROUND_CMAP)
    mesh = ax.pcolormesh(
        lon_borders, lat_borders, data,
        cmap=background_cmap, vmin=_PLOT_SPEED_VMIN, vmax=_PLOT_SPEED_VMAX,
        transform=ccrs.PlateCarree(), zorder=0,
    )
    if colorbar_rect is not None:
        colorbar = plt.colorbar(mesh, cax=fig.add_axes(colorbar_rect))
    else:
        colorbar = plt.colorbar(mesh, ax=ax)
    colorbar.set_label(PLOT_BACKGROUND_LABELS[_PLOT_BACKGROUND_VARIABLE], fontsize=PLOT_TITLE_FONTSIZE)
    colorbar.ax.tick_params(labelsize=PLOT_TITLE_FONTSIZE)
    coastline_color = PLOT_COASTLINE_COLOR_DARK if _PLOT_DARK_MODE else PLOT_COASTLINE_COLOR
    if _PLOT_COASTLINE == "GSHHFS":
        ax.add_feature(
            cfeature.GSHHSFeature(scale="full", edgecolor=coastline_color, facecolor="none"),
            zorder=1,
        )
    elif _PLOT_COASTLINE == "cartopy10m":
        ax.coastlines(resolution="10m", color=coastline_color, zorder=1)
    elif _PLOT_COASTLINE == "cartopy50m":
        ax.coastlines(resolution="50m", color=coastline_color, zorder=1)
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
    if _PLOT_LOGO is not None:
        logo_ax = ax.inset_axes(
            (
                1 - PLOT_LOGO_PADDING - PLOT_LOGO_SIZE,
                PLOT_LOGO_PADDING,
                PLOT_LOGO_SIZE,
                PLOT_LOGO_SIZE,
            ),
            zorder=PLOT_LOGO_ZORDER,
        )
        logo_ax.imshow(_PLOT_LOGO)
        # The image keeps its aspect ratio, so it shrinks inside the box
        # along one direction: keep it glued to the bottom right corner.
        logo_ax.set_anchor("SE")
        logo_ax.set_axis_off()
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    plt.savefig(output_dir / f"plot{index:0>5}.png", dpi=PLOT_DPI)
    plt.close()


def _plot_frame_star(args: tuple[int, np.datetime64, str, Path]) -> None:
    return plot_frame(*args)


def _init_plot_worker() -> None:
    """
    Initializer of the plot worker processes. The parent process may have
    already used dask's threaded scheduler (e.g. to compute the bounds of
    the colorbar): its thread pool is inherited by the forked workers, but
    without its threads, so any computation submitted to it would wait
    forever. Each worker reads only one time slice per frame, so it simply
    runs its dask computations in its own thread.
    """
    import dask

    dask.config.set(scheduler="synchronous")


def _preload_coastline(coastline: str) -> None:
    """
    Read the coastline geometries once in the parent process: cartopy caches
    them, and the forked workers inherit the cache. Otherwise each worker
    would read the whole (global) shapefile again for its first frame,
    which takes about a minute for the full resolution GSHHS one.
    """
    import cartopy.feature as cfeature

    if coastline == "GSHHFS":
        # Same feature (scale and levels) drawn by plot_frame
        feature = cfeature.GSHHSFeature(scale="full")
    elif coastline == "cartopy10m":
        feature = cfeature.NaturalEarthFeature("physical", "coastline", "10m")
    elif coastline == "cartopy50m":
        feature = cfeature.NaturalEarthFeature("physical", "coastline", "50m")
    else:
        return
    for _ in feature.geometries():
        pass


def _open_background_dataset(
        local_directory: Path, domain: Domain, date: np.datetime64, variable: str
) -> xr.Dataset:
    """
    Lazily open the data of the model (as stored by the download action)
    needed to draw the given background variable: the velocities for
    "speed", or the file of the variable itself otherwise.
    """
    file_variable = "velocities" if variable == "speed" else variable
    path = get_data_file_path(
        file_variable, local_directory, domain=domain, date=date,
        create_parents=False,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Data of the model not found: {path}. Run `comertine download "
            f"-d {domain.name} -t {np.datetime_as_string(date, unit='D')} "
            f"-l {local_directory}` first."
        )
    return xr.open_zarr(path, chunks={})


def plot_particles(
        input_dir: Path,
        local_path: Path,
        output_dir: str,
        static_dir: Path | None = None,
        land_filler: str = "none",
        coastline: str = "GSHHFS",
        background_variable: str = "speed",
        background_cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        tight_layout: bool = False,
        dark_mode: bool = False,
        show_progress: bool = True,
):
    global _PARTICLE_DATASETS, _PLOT_EXTENT, _PLOT_DS, _PLOT_BACKGROUND_VARIABLE, _PLOT_SPEED_VMIN, _PLOT_SPEED_VMAX, _PLOT_HILLSHADE, _PLOT_COASTLINE
    global _PLOT_TIGHT_LAYOUT, _PLOT_DARK_MODE, _PLOT_LOGO
    global PLOT_GRIDLINE_STEP, PLOT_BACKGROUND_CMAP

    _PLOT_COASTLINE = coastline
    _PLOT_BACKGROUND_VARIABLE = background_variable
    PLOT_BACKGROUND_CMAP = background_cmap
    _PLOT_SPEED_VMIN = vmin
    _PLOT_SPEED_VMAX = vmax
    _PLOT_DARK_MODE = dark_mode

    if PLOT_LOGO_PATH.exists():
        import matplotlib.image as mpimg
        _PLOT_LOGO = mpimg.imread(PLOT_LOGO_PATH)
    else:
        warnings.warn(f"Logo file {PLOT_LOGO_PATH} not found: no logo will be drawn")

    simulation_file = input_dir / "simulation.json"
    simulation = LagrangianSimulation.model_validate_json(simulation_file.read_text())

    domain = simulation.domain
    date = np.datetime64(simulation.date, 's')

    need_dem = land_filler != "none"
    if need_dem or coastline != "none":
        if static_dir is None:
            raise ValueError(
                "A static data directory is required to draw the land "
                "filler or the coastline"
            )
        # Fails early (before reading any model data) if something is
        # missing, instead of letting cartopy try to download it.
        # configure_cartopy is called here, in the parent process, so that
        # the forked workers inherit cartopy's data_dir.
        check_static_data(
            static_dir, domain=domain, need_dem=need_dem, coastline=coastline
        )
        configure_cartopy(static_dir)
        _preload_coastline(coastline)

    output_dir_path = Path(concretize_template_string(output_dir, domain=domain, date=date))

    ds = _open_background_dataset(local_path, domain, date, background_variable)
    # Resampled on the time step of the simulation, so that each frame
    # matches a time at which the positions of the particles are saved
    ds = resample_ds_in_time(ds, simulation.config.time_interval)
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

    if tight_layout:
        _PLOT_TIGHT_LAYOUT = _compute_tight_layout(_PLOT_EXTENT)

    if domain.resolution == "regional":
        PLOT_GRIDLINE_STEP = REGIONAL_PLOT_GRIDLINE_STEP
    elif domain.resolution == "vhr":
        PLOT_GRIDLINE_STEP = VHR_PLOT_GRIDLINE_STEP
    else:
        error_msg = f"Unknown domain resolution: {domain.resolution}"
        raise ValueError(error_msg)

    if need_dem:
        dem_tile_paths = get_dem_tile_paths(
            static_dir, get_dem_product(domain), _PLOT_EXTENT
        )
        if land_filler == "hillshade":
            _PLOT_HILLSHADE = compute_hillshade(_PLOT_EXTENT, dem_tile_paths)
        elif land_filler == "colored-hillshade":
            _PLOT_HILLSHADE = compute_colored_hillshade(_PLOT_EXTENT, dem_tile_paths)
        if _PLOT_HILLSHADE is None:
            warnings.warn(f"No land in domain {domain.name}: no hillshade will be drawn")

    if _PLOT_SPEED_VMIN is None or _PLOT_SPEED_VMAX is None:
        # Computed lazily through xarray/dask (a chunked reduction), so this
        # never materializes the full background field in memory at once.
        background_field = _select_background_field(ds, _PLOT_BACKGROUND_VARIABLE)
        field_p05, field_p95 = background_field.quantile([0.05, 0.95]).compute().values
        if _PLOT_SPEED_VMIN is None:
            _PLOT_SPEED_VMIN = float(field_p05)
        if _PLOT_SPEED_VMAX is None:
            _PLOT_SPEED_VMAX = float(field_p95)

    # Loaded eagerly (and the files closed) here in the parent process, so
    # that forked workers each get plain in-memory datasets via
    # copy-on-write instead of sharing lazily-opened file handles.
    particle_datasets = []
    for f in input_dir.glob("particles_*.nc"):
        with xr.open_dataset(f) as particles:
            particle_datasets.append(particles.load())
    _PARTICLE_DATASETS = particle_datasets

    output_dir_path.mkdir(parents=True, exist_ok=True)

    varname = _get_background_variable_name(ds, _PLOT_BACKGROUND_VARIABLE)
    title_template = TITLE.replace("${DOMAIN}", domain.long_name)
    title_template = concretize_template_string(title_template, domain=domain, date=date)
    title_template = title_template.replace("${VARNAME}", varname)
    frames = []
    for i, t in enumerate(time_values):
        rounded_time = (
            (t + np.timedelta64(30, "m"))
            .astype("datetime64[h]")
            .astype("datetime64[s]")
            .astype(object)
        )
        title = title_template.replace("${TIME}", rounded_time.strftime("%Y-%m-%d %H:%M"))
        frames.append((i, t, title, output_dir_path))
    with mp.get_context("fork").Pool(initializer=_init_plot_worker) as pool, tqdm(
        total=len(frames), desc="Plotting frames", disable=not show_progress
    ) as pbar:
        for _ in pool.imap_unordered(_plot_frame_star, frames):
            pbar.update(1)




def main():
    args = read_cli_args(DOMAINS.keys())
    if args.action == "download":
        domain = DOMAINS.get(args.domain)
        simulation_date = args.date
        if simulation_date is None:
            simulation_date = np.datetime64("today", "s")
        download_data(domain=domain, date=simulation_date, local_directory=args.local_path, variables=["thetao", "so", "uo", "vo"])
    elif args.action == "download-static":
        if args.domain is None:
            domains = list(DOMAINS.values())
        else:
            domains = [DOMAINS[d] for d in args.domain]
        download_static_data(
            static_dir=args.local_path, domains=domains,
            show_progress=args.progress,
        )
    elif args.action == "compute":
        domain = DOMAINS.get(args.domain)
        simulation_date = args.date
        if simulation_date is None:
            simulation_date = np.datetime64("today", "s")

        speed_factor = args.increased_speed_factor
        if speed_factor == "auto":
            speed_factor = auto_speed_factor(domain)

        config_args = {
            "time_interval": args.time_interval,
            "batch_time_length": args.batch_time_length,
            "batch_every": args.batch_every,
            "batch_size": args.batch_size,
            "increased_speed_factor": speed_factor,
            "seed": args.seed,
        }
        config = ParticleConfig(
            **{k: v for k, v in config_args.items() if v is not None}
        )
        input_dir = args.local_path
        output_dir = Path(
            concretize_template_string(
                args.output_dir,
                domain=domain,
                date=simulation_date,
            )
        )
        cluster_args = {
            "n_workers": args.n_workers,
            "memory_limit": args.memory_limit,
        }
        cluster_args = {k: v for k, v in cluster_args.items() if v is not None}

        # The scheduler is chosen here: compute_particles runs its tasks on
        # whatever dask scheduler is active (e.g. replace this block with
        # `with dask.config.set(scheduler="processes"):` to avoid the
        # distributed cluster)
        with LocalCluster(threads_per_worker=1, **cluster_args) as cluster, Client(cluster):
            compute_particles(
                domain=domain,
                date=simulation_date,
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                show_progress=args.progress
            )
    elif args.action == "plot":
        plot_particles(
            input_dir=args.input_dir,
            local_path=args.local_path,
            output_dir=args.output_dir,
            static_dir=args.static_dir,
            land_filler=args.land_filler, coastline=args.coastline,
            background_variable=args.background_variable,
            background_cmap=args.background_cmap,
            vmin=args.vmin, vmax=args.vmax,
            tight_layout=args.tight_layout, dark_mode=args.dark_mode,
            show_progress=args.progress,
        )
    else:
        error_msg = f"Invalid action: {args.action}"
        raise ValueError(error_msg)


if __name__ == "__main__":
    main()
