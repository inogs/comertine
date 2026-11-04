import multiprocessing as mp
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import NoReturn

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
from comertine.tools.plot_config import FigureConfig
from comertine.tools.plot_config import PlotConfig
from comertine.tools.plot_config import read_plot_config
from comertine.tools.plots import make_projection
from comertine.tools.plots import resolve_colormap
from comertine.tools.static_data import check_static_data
from comertine.tools.static_data import configure_cartopy
from comertine.tools.static_data import download_static_data
from comertine.tools.static_data import get_dem_product
from comertine.tools.static_data import get_dem_tile_paths


@dataclass(frozen=True)
class _PlotContext:
    """
    Everything the plot workers need to draw a frame.

    It is built by plot_particles() in the parent process and handed to the
    workers by the initializer of the pool: since the workers are forked,
    they inherit it (copy-on-write) instead of receiving a pickled copy.
    """
    config: PlotConfig

    # The particle datasets, already loaded in memory
    particle_datasets: list[xr.Dataset]

    # The (resampled) ocean dataset used to draw the background field. Kept
    # lazy (dask-backed): each worker reads only the time slice it needs.
    ds: xr.Dataset

    # The map extent (lon_min, lon_max, lat_min, lat_max)
    extent: tuple[float, float, float, float]

    # The bounds of the colorbar of the background field (from the
    # configuration or, when not set there, from the 5th/95th percentile of
    # the values of the field)
    vmin: float
    vmax: float

    # The distance between two gridlines, for the resolution of the domain
    gridline_step: float

    # The precomputed land image (either a grayscale hillshade or an
    # already-colored hypsometric-tint hillshade) drawn under the ocean
    # field (whose NaN-over-land cells let it show through), as
    # (image, lon, lat, cmap) - cmap is None when image is already RGBA;
    # None when there is no land filler
    hillshade: tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, str | None] | None

    # Figure geometry used with the tight layout, as
    # (figure_width_px, map_axes_rect, colorbar_axes_rect), where the rects
    # are (left, bottom, width, height) in figure coordinates; None when the
    # tight layout is disabled
    tight_layout: tuple[int, tuple[float, float, float, float], tuple[float, float, float, float]] | None

    # The logo image (as returned by matplotlib.image.imread), or None if
    # no logo has to be drawn
    logo: np.typing.NDArray | None

    # The geometries of the coastline (in lon/lat), already clipped around
    # the extent of the map; None when no coastline has to be drawn
    coastline: tuple[Any, ...] | None


# Set only inside the plot workers, by _init_plot_worker()
_PLOT_CONTEXT: _PlotContext | None = None


def _load_dem_elevation(
        extent: tuple[float, float, float, float], tile_paths: list[Path],
        figure: FigureConfig,
):
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
        # The file is closed as soon as the (small) clipped piece has been
        # read: a rasterio dataset left open until the interpreter shuts down
        # makes Python print "Error in sys.excepthook" at exit
        with rioxarray.open_rasterio(tile_path) as tile:
            tile = tile.squeeze("band", drop=True)
            try:
                pieces.append(
                    tile.rio.clip_box(
                        minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max
                    ).load()
                )
            except NoDataInBounds:
                continue
    if not pieces:
        return None

    # Downsample the DEM (averaging the pixels) so that it is not larger
    # than the frame: the map always fits inside the width x height of the
    # figure keeping its aspect ratio, so scaling the DEM by
    # min(W / nx, H / ny) still leaves at least one DEM pixel for each
    # pixel of the map. Without this, the hillshade of a regional domain at
    # 90 m is wider than the whole 4K frame and needs several GB of memory.
    # Each tile is downsampled on its own, before merging them, so that the
    # full resolution grid of the whole domain is never built.
    res_x, res_y = (abs(r) for r in pieces[0].rio.resolution())
    full_nx = (lon_max - lon_min) / res_x
    full_ny = (lat_max - lat_min) / res_y
    scale = min(figure.width_px / full_nx, figure.height_px / full_ny)
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


def compute_hillshade(
        extent: tuple[float, float, float, float], tile_paths: list[Path],
        config: PlotConfig,
):
    """
    Compute a grayscale hillshade of the land for the given
    (lon_min, lon_max, lat_min, lat_max) extent.

    Returns (hillshade, lon, lat, cmap) - hillshade has shape
    (len(lat), len(lon)) with values in [0, 1], to be drawn with cmap
    "gray".
    """
    from matplotlib.colors import LightSource

    dem = _load_dem_elevation(extent, tile_paths, config.figure)
    if dem is None:
        return None
    elevation, lon, lat, dx, dy = dem

    light_source = LightSource(
        azdeg=config.land.hillshade_azimuth,
        altdeg=config.land.hillshade_altitude,
    )
    hillshade = light_source.hillshade(elevation, dx=dx, dy=dy)

    return hillshade, lon, lat, "gray"


def compute_colored_hillshade(
        extent: tuple[float, float, float, float], tile_paths: list[Path],
        config: PlotConfig,
):
    """
    Compute a hypsometric-tint hillshade of the land (elevation-based
    color, shaded by relief) for the given
    (lon_min, lon_max, lat_min, lat_max) extent.

    Returns (image, lon, lat, cmap) - image is an (len(lat), len(lon), 4)
    RGBA array already colored, so cmap is None (nothing left to map).
    """
    from matplotlib import colormaps
    from matplotlib.colors import LightSource, LinearSegmentedColormap

    dem = _load_dem_elevation(extent, tile_paths, config.figure)
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
        azdeg=config.land.hillshade_azimuth,
        altdeg=config.land.hillshade_altitude,
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
        extent: tuple[float, float, float, float], config: PlotConfig
) -> tuple[int, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """
    Compute the figure geometry used by the tight layout: the figure keeps
    its height and its width is derived from the aspect ratio of the map (in
    projected coordinates) once `extent` is set, plus the margins of
    `config.tight_layout`.

    Returns (figure_width_px, map_axes_rect, colorbar_axes_rect), with the
    rects as (left, bottom, width, height) in figure coordinates.
    """
    import cartopy.crs as ccrs
    # A bare Figure (not pyplot) so that no GUI backend gets initialized in
    # the parent process before the workers are forked.
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_axes((0, 0, 1, 1), projection=make_projection(config.projection, Extent(*extent)))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    layout = config.tight_layout
    height_px = config.figure.height_px

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    map_height_px = height_px - layout.margin_top_px - layout.margin_bottom_px
    map_width_px = map_height_px * aspect_ratio
    figure_width_px = (
        layout.margin_left_px + map_width_px + layout.colorbar_pad_px
        + layout.colorbar_width_px + layout.margin_right_px
    )
    # Rounded up to a multiple of `width_multiple` pixels, so that
    # the frames are friendly to video encoders (chroma subsampling and
    # macroblocks) when turning them into a movie.
    figure_width_px = int(
        np.ceil(figure_width_px / layout.width_multiple) * layout.width_multiple
    )

    map_rect = (
        layout.margin_left_px / figure_width_px,
        layout.margin_bottom_px / height_px,
        map_width_px / figure_width_px,
        map_height_px / height_px,
    )
    colorbar_rect = (
        (layout.margin_left_px + map_width_px + layout.colorbar_pad_px) / figure_width_px,
        layout.margin_bottom_px / height_px,
        layout.colorbar_width_px / figure_width_px,
        map_height_px / height_px,
    )
    return figure_width_px, map_rect, colorbar_rect


def plot_frame(index: int, current_time: np.datetime64, title: str, output_dir: Path) -> None:
    """
    Plot the particles (inside_domain only) at `current_time`, gathering
    them from every batch in `context.particle_datasets` that covers that time,
    and save the result to output_dir / f"plot{index}.png".

    `title` is the fully resolved plot title for this frame, and
    `output_dir` the directory to save it into; both computed by the
    caller.

    Runs in a worker process forked from plot_particles(): everything else
    it needs (the configuration, the datasets, ...) is in `_PLOT_CONTEXT`,
    inherited from the parent rather than passed as an argument.
    """
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    context = _PLOT_CONTEXT
    if context is None:
        raise Exception("The plot context has not been initialized")
    config = context.config

    particle_rgb = to_rgba(config.particles.color)[:3]
    trail_weights = config.trail.weights()

    latitudes = []
    longitudes = []
    alphas = []
    trail_segments = []
    trail_colors = []


    for particles in context.particle_datasets:
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
                return config.particles.alpha * 4 * k
            if k > 0.75:
                return config.particles.alpha * (1 - k) * 4
            return config.particles.alpha

        frame = particles.isel(time=time_index)
        inside = frame.inside_domain.values
        n_particles = int(inside.sum())

        latitudes.append(frame.latitude.values[inside])
        longitudes.append(frame.longitude.values[inside])
        alphas.append(np.full(n_particles, batch_fade_alpha(current_time)))

        # --- trail ---
        window_start = max(0, time_index - config.trail.length + 1)
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
        # window left more than config.trail.length frames ago.
        reversed_inside = window_inside[::-1, :]
        has_any_true = reversed_inside.any(axis=0)
        frames_since_exit = np.argmax(reversed_inside, axis=0)

        draw_mask = has_any_true & (frames_since_exit <= config.trail.fade_out_frames)
        if not np.any(draw_mask):
            continue

        exit_fade = np.where(
            frames_since_exit == 0,
            1.0,
            np.clip(
                1.0 - frames_since_exit / (config.trail.fade_out_frames + 1), 0.0, 1.0
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
            [trail_weights[d] for d in segment_distance]
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

    ds_time_values = context.ds.time.values
    ds_time_index = int(np.searchsorted(ds_time_values, current_time))

    background_field = _select_background_field(context.ds, config.background.variable)

    lat_borders, _ = get_borders_and_sizes(context.ds.latitude.values)
    lon_borders, _ = get_borders_and_sizes(context.ds.longitude.values)

    if config.figure.dark_mode:
        plt.style.use("dark_background")

    if context.tight_layout is not None:
        figure_width_px, map_rect, colorbar_rect = context.tight_layout
    else:
        figure_width_px, map_rect, colorbar_rect = config.figure.width_px, None, None
    fig = plt.figure(
        figsize=(figure_width_px / config.figure.dpi, config.figure.height_px / config.figure.dpi),
        dpi=config.figure.dpi,
    )
    projection = make_projection(config.projection, Extent(*context.extent))
    if map_rect is not None:
        ax = fig.add_axes(map_rect, projection=projection)
    else:
        ax = plt.axes(projection=projection)
    ax.set_extent(context.extent, crs=ccrs.PlateCarree())
    if context.hillshade is not None:
        hillshade_image, hillshade_lon, hillshade_lat, hillshade_cmap = context.hillshade
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
            transform=ccrs.PlateCarree(), zorder=config.land.zorder,
        )

    data = background_field.isel(time=ds_time_index).values
    background_cmap = resolve_colormap(config.background.cmap)
    mesh = ax.pcolormesh(
        lon_borders, lat_borders, data,
        cmap=background_cmap, vmin=context.vmin, vmax=context.vmax,
        transform=ccrs.PlateCarree(), zorder=0,
    )
    if colorbar_rect is not None:
        colorbar = plt.colorbar(mesh, cax=fig.add_axes(colorbar_rect))
    else:
        colorbar = plt.colorbar(mesh, ax=ax)
    colorbar.set_label(config.background.labels[config.background.variable], fontsize=config.figure.title_fontsize)
    colorbar.ax.tick_params(labelsize=config.figure.title_fontsize)
    if context.coastline is not None:
        coastline_color = config.coastline.color_dark if config.figure.dark_mode else config.coastline.color
        ax.add_geometries(
            context.coastline, crs=ccrs.PlateCarree(),
            edgecolor=coastline_color, facecolor="none", zorder=1,
        )
    if trail_segments:
        trail_collection = LineCollection(
            np.concatenate(trail_segments, axis=0),
            colors=np.concatenate(trail_colors, axis=0),
            linewidths=config.trail.linewidth,
            transform=ccrs.PlateCarree(), zorder=config.trail.zorder,
        )
        ax.add_collection(trail_collection)
    ax.scatter(
        longitudes, latitudes, s=config.particles.size, marker='.', linewidths=0,
        color=config.particles.color, alpha=alphas,
        transform=ccrs.PlateCarree(), zorder=2,
    )

    gridlines = ax.gridlines(
        draw_labels=True, color="lightgray", linestyle="--", zorder=1,
        xlocs=mticker.MultipleLocator(context.gridline_step),
        ylocs=mticker.MultipleLocator(context.gridline_step),
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": config.gridlines.fontsize}
    gridlines.ylabel_style = {"size": config.gridlines.fontsize}
    if context.logo is not None:
        logo_ax = ax.inset_axes(
            (
                1 - config.logo.padding - config.logo.size,
                config.logo.padding,
                config.logo.size,
                config.logo.size,
            ),
            zorder=config.logo.zorder,
        )
        logo_ax.imshow(context.logo)
        # The image keeps its aspect ratio, so it shrinks inside the box
        # along one direction: keep it glued to the bottom right corner.
        logo_ax.set_anchor("SE")
        logo_ax.set_axis_off()
    ax.set_title(title, fontsize=config.figure.title_fontsize)
    plt.savefig(output_dir / f"plot{index:0>5}.png", dpi=config.figure.dpi)
    plt.close()


def _plot_frame_star(args: tuple[int, np.datetime64, str, Path]) -> None:
    return plot_frame(*args)


def _init_plot_worker(context: _PlotContext) -> None:
    """
    Initializer of the plot worker processes.

    It stores `context` in `_PLOT_CONTEXT`, where plot_frame() reads it. The
    workers are forked, so `context` is inherited, not pickled.

    The parent process may have already used dask's threaded scheduler
    (e.g. to compute the bounds of the colorbar): its thread pool is
    inherited by the forked workers, but without its threads, so any
    computation submitted to it would wait forever. Each worker reads only
    one time slice per frame, so it simply runs its dask computations in
    its own thread.

    The frames are only saved to files, so the workers use the
    non-interactive Agg backend of matplotlib. Otherwise, when a display is
    available, matplotlib picks a GUI backend (e.g. TkAgg), which is not
    safe in forked processes: the workers print errors from tkinter and can
    hang forever.
    """
    import dask
    import matplotlib

    global _PLOT_CONTEXT
    _PLOT_CONTEXT = context

    matplotlib.use("Agg")
    dask.config.set(scheduler="synchronous")


# How much the box used to clip the coastline is larger than the extent of
# the map, as a fraction of the largest side of the extent (see
# _load_coastline)
COASTLINE_CLIP_MARGIN = 0.5


def _load_coastline(
        coastline: str, extent: tuple[float, float, float, float]
) -> tuple[Any, ...] | None:
    """
    Read the geometries of the coastline and clip them around `extent`.

    This runs once, in the parent process. The coastline datasets are
    global: the full resolution GSHHS one contains, for example, the whole
    Eurasia as a single polygon with millions of points. Drawing it directly
    would make every worker project the whole polygon for its first frame
    (minutes); the clipped geometries, instead, are small.

    The clipping box is larger than `extent` (see COASTLINE_CLIP_MARGIN): a
    polygon clipped by a box gets new edges along the box, and they must
    stay outside the visible part of the map, also for the projections
    (e.g. NearsidePerspective) that show a bit more than `extent`.

    Args:
        coastline: The kind of coastline ("GSHHFS", "cartopy10m",
            "cartopy50m" or "none").
        extent: The extent of the map (lon_min, lon_max, lat_min, lat_max).

    Returns:
        The clipped geometries (in lon/lat), or None for "none".
    """
    import cartopy.feature as cfeature
    import shapely

    if coastline == "GSHHFS":
        feature = cfeature.GSHHSFeature(scale="full")
    elif coastline == "cartopy10m":
        feature = cfeature.NaturalEarthFeature("physical", "coastline", "10m")
    elif coastline == "cartopy50m":
        feature = cfeature.NaturalEarthFeature("physical", "coastline", "50m")
    else:
        return None

    lon_min, lon_max, lat_min, lat_max = extent
    margin = COASTLINE_CLIP_MARGIN * max(lon_max - lon_min, lat_max - lat_min)
    clip_extent = (
        lon_min - margin, lon_max + margin,
        max(lat_min - margin, -90.), min(lat_max + margin, 90.),
    )
    clip_box = shapely.box(
        clip_extent[0], clip_extent[2], clip_extent[1], clip_extent[3]
    )

    clipped = []
    for geometry in feature.intersecting_geometries(clip_extent):
        piece = geometry.intersection(clip_box)
        if not piece.is_empty:
            clipped.append(piece)
    return tuple(clipped)


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
        config: PlotConfig | None = None,
        show_progress: bool = True,
):
    """
    Draw one PNG frame for each time step of the simulation in `input_dir`.

    Args:
        input_dir: The directory with the output of the compute action.
        local_path: The directory with the data of the model.
        output_dir: Where the frames are saved; ${DOMAIN} and ${DATE} are
            replaced with the domain and the date of the simulation.
        static_dir: The directory with the static data (DEM and
            coastlines); required unless both the land filler and the
            coastline are "none".
        config: The configuration of the plots (default: PlotConfig()).
        show_progress: Whether to show a progress bar.
    """
    if config is None:
        config = PlotConfig()

    logo = None
    if config.logo.path is not None:
        if not config.logo.path.exists():
            raise FileNotFoundError(f"Logo file {config.logo.path} not found")
        import matplotlib.image as mpimg
        logo = mpimg.imread(config.logo.path)

    simulation_file = input_dir / "simulation.json"
    simulation = LagrangianSimulation.model_validate_json(simulation_file.read_text())

    domain = simulation.domain
    date = np.datetime64(simulation.date, 's')

    land_filler = config.land.filler
    coastline = config.coastline.kind
    background_variable = config.background.variable

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

    output_dir_path = Path(concretize_template_string(output_dir, domain=domain, date=date))

    ds = _open_background_dataset(local_path, domain, date, background_variable)
    # Resampled on the time step of the simulation, so that each frame
    # matches a time at which the positions of the particles are saved.
    # Kept lazy (dask-backed) and shared as-is with the forked workers: the
    # dataset can be hundreds of GB, so each worker reads only the time
    # slice it needs from disk instead of the whole thing being loaded into
    # memory here.
    ds = resample_ds_in_time(ds, simulation.config.time_interval)
    time_values = ds.time.values

    extent = (
        float(ds.longitude.min()), float(ds.longitude.max()),
        float(ds.latitude.min()), float(ds.latitude.max()),
    )

    tight_layout = None
    if config.tight_layout.enabled:
        tight_layout = _compute_tight_layout(extent, config)

    coastline_geometries = _load_coastline(coastline, extent)

    hillshade = None
    if need_dem:
        dem_tile_paths = get_dem_tile_paths(
            static_dir, get_dem_product(domain), extent
        )
        if land_filler == "hillshade":
            hillshade = compute_hillshade(extent, dem_tile_paths, config)
        elif land_filler == "colored-hillshade":
            hillshade = compute_colored_hillshade(extent, dem_tile_paths, config)
        if hillshade is None:
            warnings.warn(f"No land in domain {domain.name}: no hillshade will be drawn")

    vmin = config.background.vmin
    vmax = config.background.vmax
    if vmin is None or vmax is None:
        # Computed lazily through xarray/dask (a chunked reduction), so this
        # never materializes the full background field in memory at once.
        background_field = _select_background_field(ds, background_variable)
        field_p05, field_p95 = background_field.quantile([0.05, 0.95]).compute().values
        if vmin is None:
            vmin = float(field_p05)
        if vmax is None:
            vmax = float(field_p95)

    # Loaded eagerly (and the files closed) here in the parent process, so
    # that forked workers each get plain in-memory datasets via
    # copy-on-write instead of sharing lazily-opened file handles.
    particle_datasets = []
    for f in input_dir.glob("particles_*.nc"):
        with xr.open_dataset(f) as particles:
            particle_datasets.append(particles.load())

    context = _PlotContext(
        config=config,
        particle_datasets=particle_datasets,
        ds=ds,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        gridline_step=config.gridlines.step(domain),
        hillshade=hillshade,
        tight_layout=tight_layout,
        logo=logo,
        coastline=coastline_geometries,
    )

    output_dir_path.mkdir(parents=True, exist_ok=True)

    varname = _get_background_variable_name(ds, background_variable)
    title_template = config.figure.title_template.replace("${DOMAIN}", domain.long_name)
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
    with mp.get_context("fork").Pool(
        initializer=_init_plot_worker, initargs=(context,)
    ) as pool, tqdm(
        total=len(frames), desc="Plotting frames", disable=not show_progress
    ) as pbar:
        for _ in pool.imap_unordered(_plot_frame_star, frames):
            pbar.update(1)


def _cli_error(action: str, message: str) -> NoReturn:
    """Print an error like argparse does, and exit with status 2."""
    print(f"comertine {action}: error: {message}", file=sys.stderr)
    sys.exit(2)


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
        try:
            plot_config = read_plot_config(args.config)
        except ValueError as e:
            _cli_error("plot", f"invalid configuration file {args.config}:\n{e}")

        needs_static_data = (
            plot_config.land.filler != "none"
            or plot_config.coastline.kind != "none"
        )
        if needs_static_data and args.static_dir is None:
            _cli_error(
                "plot",
                "--static-dir is required unless both the land filler and "
                "the coastline are 'none' in the configuration",
            )

        plot_particles(
            input_dir=args.input_dir,
            local_path=args.local_path,
            output_dir=args.output_dir,
            static_dir=args.static_dir,
            config=plot_config,
            show_progress=args.progress,
        )
    else:
        error_msg = f"Invalid action: {args.action}"
        raise ValueError(error_msg)


if __name__ == "__main__":
    main()
