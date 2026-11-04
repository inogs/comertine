import cartopy.crs as ccrs
import cmocean
from matplotlib import colormaps

from comertine.tools.geometry import Extent
from comertine.tools.plot_config import ProjectionConfig


def resolve_colormap(name: str):
    """
    Resolve the name of a colormap into an actual matplotlib Colormap.

    A "cmo:" prefix selects a cmocean colormap by name (e.g.
    "cmo:thermal"); otherwise `name` is looked up among matplotlib's
    built-in colormaps.
    """
    if name.startswith("cmo:"):
        return cmocean.cm.cmap_d[name.removeprefix("cmo:")]

    return colormaps[name]


def make_projection(config: ProjectionConfig, extent: Extent):
    """
    Build the cartopy projection described by `config` (whose name is the
    name of a cartopy.crs class) used to draw the map axes, centered on
    `extent` when the projection supports it.
    """
    projection_name = config.name
    if projection_name == "LambertConformal":
        return ccrs.LambertConformal(
            central_longitude=extent.central_longitude(),
            central_latitude=extent.central_latitude(),
        )
    if projection_name == "NearsidePerspective":
        return ccrs.NearsidePerspective(
            central_longitude=extent.central_longitude(),
            central_latitude=extent.central_latitude(),
            satellite_height=config.nearside_satellite_height,
        )
    return getattr(ccrs, projection_name)()
