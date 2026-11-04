from pathlib import Path
from typing import Any
from typing import Literal
from typing import Self

import numpy as np
import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from comertine.domains import Domain


BackgroundVariable = Literal["speed", "thetao", "so", "o2", "chl", "no3"]
CoastlineKind = Literal["none", "GSHHFS", "cartopy10m", "cartopy50m"]
LandFiller = Literal["none", "hillshade", "colored-hillshade"]
ProjectionName = Literal[
    "PlateCarree", "Mercator", "LambertConformal", "NearsidePerspective"
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _merge_with_default(cls: type[BaseModel], field_name: str, value: Any) -> Any:
    """
    Update the default value (a dict) of a field with the given entries.

    In this way, a configuration file that sets only some entries of a dict
    keeps the default value of the other ones.
    """
    if value is None:
        return cls.model_fields[field_name].get_default(call_default_factory=True)
    if isinstance(value, dict):
        default = cls.model_fields[field_name].get_default(call_default_factory=True)
        return {**default, **value}
    return value


class FigureConfig(_FrozenModel):
    # Plot resolution, sized for a fullscreen 4K display
    width_px: int = Field(default=3840, gt=0)
    height_px: int = Field(default=2160, gt=0)
    dpi: int = Field(default=100, gt=0)

    # Title of each frame; ${DOMAIN}, ${VARNAME} and ${TIME} are replaced
    # with the long name of the domain, the name of the background variable
    # and the time of the frame
    title_template: str = "${DOMAIN} - ${VARNAME} - ${TIME}"
    # Also used for the label of the colorbar
    title_fontsize: float = Field(default=28, gt=0)

    # Draw the frames using matplotlib's "dark_background" style
    dark_mode: bool = False


class TightLayoutConfig(_FrozenModel):
    """
    Geometry of the figure when the tight layout is enabled.

    The figure keeps `FigureConfig.height_px` as its height, the map fills
    the space left by the top/bottom margins and the figure width is
    recomputed from the aspect ratio of the map, plus room for the latitude
    labels on the left and for the colorbar (with its tick labels and label)
    on the right.
    """
    # Recompute the width of the frames from the aspect ratio of the map
    # instead of using a fixed canvas
    enabled: bool = False

    # Margins (in pixels) around the map
    margin_left_px: int = Field(default=130, ge=0)
    margin_right_px: int = Field(default=220, ge=0)
    margin_top_px: int = Field(default=90, ge=0)
    margin_bottom_px: int = Field(default=70, ge=0)

    colorbar_pad_px: int = Field(default=50, ge=0)
    colorbar_width_px: int = Field(default=60, gt=0)

    # The recomputed figure width is rounded up to a multiple of this value
    width_multiple: int = Field(default=8, gt=0)


class ProjectionConfig(_FrozenModel):
    # Name of the cartopy.crs class used to draw the map axes. Data
    # coordinates stay in plain lon/lat (ccrs.PlateCarree()) regardless of
    # this setting.
    name: ProjectionName = "Mercator"

    # Satellite altitude (in meters) used by the "NearsidePerspective"
    # projection. Lower than the cartopy default (geostationary, ~35.8e6 m)
    # so the globe's curvature actually reads on a small domain.
    nearside_satellite_height: float = Field(default=500_000, gt=0)


class GridlinesConfig(_FrozenModel):
    # Distance (in degrees) between two gridlines, for each resolution of
    # the domains
    # (the resolutions not set in the configuration keep their default)
    step_by_resolution: dict[str, float] = {"regional": 0.5, "vhr": 0.1}
    fontsize: float = Field(default=20, gt=0)

    @field_validator("step_by_resolution", mode="before")
    @classmethod
    def _merge_steps(cls, value: Any) -> Any:
        return _merge_with_default(cls, "step_by_resolution", value)

    @field_validator("step_by_resolution", mode="after")
    @classmethod
    def _check_steps(cls, value: dict[str, float]) -> dict[str, float]:
        for resolution, step in value.items():
            if not step > 0:
                msg = f"the gridline step for {resolution!r} must be positive"
                raise ValueError(msg)
        return value

    def step(self, domain: Domain) -> float:
        """Return the gridline step for the resolution of `domain`."""
        try:
            return self.step_by_resolution[domain.resolution]
        except KeyError:
            msg = (
                f"No gridline step for resolution {domain.resolution!r} "
                f"(domain {domain.name})"
            )
            raise ValueError(msg) from None


class BackgroundConfig(_FrozenModel):
    # The field drawn as the background color field: the current speed
    # (computed from uo/vo) or one of the variables downloaded by the
    # download action: temperature (thetao), salinity (so), dissolved
    # oxygen (o2), chlorophyll (chl) or nitrate (no3). Their time resolution
    # does not matter: each frame interpolates between the two closest times
    # of the data
    variable: BackgroundVariable = "speed"

    # A matplotlib colormap name (e.g. "viridis") or, prefixed with "cmo:",
    # the name of a cmocean colormap (e.g. "cmo:thermal")
    cmap: str = "viridis"

    # Bounds of the colorbar; when None, they are computed from the data as
    # the 5th/95th percentile of the values of the background field
    vmin: float | None = None
    vmax: float | None = None

    # Label of the colorbar for each background variable (the variables not
    # set in the configuration keep their default)
    labels: dict[BackgroundVariable, str] = {
        "speed": "Speed [m/s]",
        "thetao": "Temperature [°C]",
        "so": "Salinity [PSU]",
        "o2": "Dissolved oxygen [mmol/m³]",
        "chl": "Chlorophyll [mg/m³]",
        "no3": "Nitrate [mmol/m³]",
    }

    @field_validator("labels", mode="before")
    @classmethod
    def _merge_labels(cls, value: Any) -> Any:
        return _merge_with_default(cls, "labels", value)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.variable not in self.labels:
            msg = f"No label for the background variable {self.variable!r}"
            raise ValueError(msg)
        if (
            self.vmin is not None
            and self.vmax is not None
            and self.vmin >= self.vmax
        ):
            msg = f"vmin ({self.vmin}) must be smaller than vmax ({self.vmax})"
            raise ValueError(msg)
        return self


class LandConfig(_FrozenModel):
    # What is drawn on land, below the ocean field: nothing, a grayscale
    # hillshade or a hillshade colored with a hypsometric tint
    filler: LandFiller = "none"

    hillshade_azimuth: float = 315
    hillshade_altitude: float = 45
    zorder: float = -1


class CoastlineConfig(_FrozenModel):
    # Which coastline to draw: "none" (skip it), "GSHHFS" (full-resolution
    # cfeature.GSHHSFeature), "cartopy10m"/"cartopy50m" (cartopy's own
    # Natural Earth coastlines at 10m/50m resolution)
    kind: CoastlineKind = "GSHHFS"

    # Color of the coastline, in the default style and in dark mode
    color: str = "black"
    color_dark: str = "white"


class LogoConfig(_FrozenModel):
    """
    Logo drawn inside the map axes, in the bottom right corner.

    The logo is scaled (keeping its aspect ratio) to fit a box `size` times
    the size of the map axes, placed `padding` (in axes coordinates) away
    from the bottom and right borders, so it never sticks out of the axes.
    """
    # Relative to the current working directory; if the file does not
    # exist, no logo is drawn
    path: Path | None = None
    size: float = Field(default=1 / 6, gt=0, le=1)
    padding: float = Field(default=0.01, ge=0, lt=1)
    zorder: float = 3


class ParticlesStyleConfig(_FrozenModel):
    size: float = Field(default=15, gt=0)
    color: str = "white"
    alpha: float = Field(default=0.8, ge=0, le=1)


class TrailConfig(_FrozenModel):
    # Number of past positions (current one included) shown as a trail
    # behind each particle. Truncated near the start of a particle's own
    # batch file, where fewer past positions are known.
    length: int = Field(default=24, ge=2)

    # Per-segment weight of the trail, ordered from the newest (closest to
    # the current position) to the oldest; it must have `length - 1`
    # entries. When None, a quadratic decay from 1 to 0 is used.
    segment_weights: tuple[float, ...] | None = None

    # How many frames after leaving the domain a particle's trail keeps
    # fading out before disappearing entirely (its position is frozen once
    # it exits, so no new segments are added during this window)
    fade_out_frames: int = Field(default=2, ge=0)

    linewidth: float = Field(default=2.0, gt=0)
    zorder: float = 1.5

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        if (
            self.segment_weights is not None
            and len(self.segment_weights) != self.length - 1
        ):
            msg = (
                f"segment_weights must have length - 1 = {self.length - 1} "
                f"entries (received {len(self.segment_weights)})"
            )
            raise ValueError(msg)
        return self

    def weights(self) -> np.ndarray:
        """Return the weight of each segment of the trail."""
        if self.segment_weights is not None:
            return np.asarray(self.segment_weights, dtype=np.float64)
        return np.linspace(1, 0, self.length - 1) ** 2


class PlotConfig(_FrozenModel):
    figure: FigureConfig = FigureConfig()
    tight_layout: TightLayoutConfig = TightLayoutConfig()
    projection: ProjectionConfig = ProjectionConfig()
    gridlines: GridlinesConfig = GridlinesConfig()
    background: BackgroundConfig = BackgroundConfig()
    land: LandConfig = LandConfig()
    coastline: CoastlineConfig = CoastlineConfig()
    logo: LogoConfig = LogoConfig()
    particles: ParticlesStyleConfig = ParticlesStyleConfig()
    trail: TrailConfig = TrailConfig()

    @model_validator(mode="before")
    @classmethod
    def _empty_sections(cls, data: Any) -> Any:
        # A section written in the YAML file without any option (e.g. just
        # "figure:") is read as None: it means "use the defaults"
        if isinstance(data, dict):
            return {k: ({} if v is None else v) for k, v in data.items()}
        return data


def read_plot_config(config_file: Path | None) -> PlotConfig:
    """
    Read the configuration of the plots from a YAML file.

    The file has the same structure of `PlotConfig`: one section for each of
    its fields (figure, tight_layout, projection, ...), with the options of
    the corresponding sub-configuration. Every option, and every section, can
    be omitted: in that case, its default value is used. An empty file (or
    no file at all) gives the default configuration.

    Args:
        config_file: The path of the YAML file, or None to use the defaults.

    Returns:
        The configuration of the plots.

    Raises:
        ValueError: if the file is not a valid configuration.
    """
    if config_file is None:
        return PlotConfig()

    with open(config_file, encoding="utf-8") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"not a valid YAML file ({e})") from e

    if content is None:
        content = {}
    if not isinstance(content, dict):
        msg = (
            "the file must contain a mapping of sections (figure, "
            "tight_layout, projection, ...)"
        )
        raise ValueError(msg)

    return PlotConfig.model_validate(content)
