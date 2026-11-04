import cartopy.crs as ccrs
import cmocean
from matplotlib import colormaps

from comertine.tools.geometry import Extent


# Satellite altitude (in meters) used by the "NearsidePerspective"
# projection. Lower than the cartopy default (geostationary, ~35.8e6 m) so
# the globe's curvature actually reads on a small domain.
PLOT_NEARSIDE_SATELLITE_HEIGHT = 500_000


def resolve_colormap(name: str):
    """
    Resolve a --background-cmap string into an actual matplotlib Colormap.

    A "cmo:" prefix selects a cmocean colormap by name (e.g.
    "cmo:thermal"); otherwise `name` is looked up among matplotlib's
    built-in colormaps.
    """
    if name.startswith("cmo:"):
        return cmocean.cm.cmap_d[name.removeprefix("cmo:")]

    return colormaps[name]


def make_projection(projection_name: str, extent: Extent):
    """
    Build the cartopy projection named `projection_name` (the name of a
    cartopy.crs class) used to draw the map axes, centered on `extent` when
    the projection supports it.
    """
    if projection_name == "LambertConformal":
        return ccrs.LambertConformal(
            central_longitude=extent.central_longitude(),
            central_latitude=extent.central_latitude(),
        )
    if projection_name == "NearsidePerspective":
        return ccrs.NearsidePerspective(
            central_longitude=extent.central_longitude(),
            central_latitude=extent.central_latitude(),
            satellite_height=PLOT_NEARSIDE_SATELLITE_HEIGHT,
        )
    return getattr(ccrs, projection_name)()
