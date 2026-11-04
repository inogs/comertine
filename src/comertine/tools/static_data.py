"""
Static data (i.e., data that does not depend on the simulation date) needed
by comertine, and the functions to download it.

This is kept separate from the rest of the code so that it can be run on a
machine with internet access (e.g., the login node of a cluster) before
running the other actions on nodes that may be offline.

All the static data lives inside a single directory chosen by the user
(the "static directory"), with the following layout:

    <static_dir>/
        dem/GLO-30/            Copernicus DEM tiles at 30 m (vhr domains)
            tileList.txt       list of all the tiles of the product
            Copernicus_DSM_COG_10_<lat>_00_<lon>_00_DEM.tif
        dem/GLO-90/            Copernicus DEM tiles at 90 m (regional
                               domains), same layout as above
        cartopy/               cartopy's data_dir (GSHHS and Natural Earth
                               coastlines)
"""
import math
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from comertine.domains import Domain


class DemProduct(BaseModel):
    """
    A Copernicus DEM product, published as 1x1 degree tiles on a public,
    no-auth AWS Open Data bucket.
    """
    name: str
    bucket_url: str
    # Code of the resolution inside the tile names ("10" = 1 arcsec,
    # "30" = 3 arcsec)
    resolution_code: str


GLO_30 = DemProduct(
    name="GLO-30",
    bucket_url="https://copernicus-dem-30m.s3.amazonaws.com",
    resolution_code="10",
)
GLO_90 = DemProduct(
    name="GLO-90",
    bucket_url="https://copernicus-dem-90m.s3.amazonaws.com",
    resolution_code="30",
)

# Which DEM product is used for each domain resolution: 30 m is overkill
# (and gigabytes of data) for the regional domains
DEM_PRODUCTS = {
    "vhr": GLO_30,
    "regional": GLO_90,
}

DEM_SUBDIR = "dem"
# File, published in each bucket, listing all the existing tiles: the tiles
# that would only contain sea do not exist
DEM_TILE_LIST_FILE_NAME = "tileList.txt"

CARTOPY_SUBDIR = "cartopy"

# Coastline datasets used by the plot (see the --coastline option).
# GSHHS: cartopy's GSHHSFeature(scale="full") reads scale "f", level 1
GSHHS_SCALE = "f"
GSHHS_LEVEL = 1
# Natural Earth: resolutions used by ax.coastlines(resolution=...)
NATURAL_EARTH_COASTLINE_RESOLUTIONS = ("10m", "50m")


def get_dem_product(domain: Domain) -> DemProduct:
    """The DEM product used for the given domain."""
    return DEM_PRODUCTS[domain.resolution]


def get_dem_dir(static_dir: Path, product: DemProduct) -> Path:
    """Directory that holds the tiles of a DEM product."""
    return Path(static_dir) / DEM_SUBDIR / product.name


def domain_extent(domain: Domain) -> tuple[float, float, float, float]:
    """The (lon_min, lon_max, lat_min, lat_max) extent of a domain."""
    bbox = domain.bounding_box
    return (
        bbox.minimum_longitude, bbox.maximum_longitude,
        bbox.minimum_latitude, bbox.maximum_latitude,
    )


def _tile_name(product: DemProduct, lat: int, lon: int) -> str:
    """Name of the tile whose south-west corner is (lat, lon)."""
    lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
    lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"
    return f"Copernicus_DSM_COG_{product.resolution_code}_{lat_str}_{lon_str}_DEM"


def _tile_names_for_extent(
        product: DemProduct, extent: tuple[float, float, float, float]
) -> list[str]:
    """Names of the 1x1 degree tiles that intersect the given extent."""
    lon_min, lon_max, lat_min, lat_max = extent
    lons = range(math.floor(lon_min), max(math.ceil(lon_max), math.floor(lon_min) + 1))
    lats = range(math.floor(lat_min), max(math.ceil(lat_max), math.floor(lat_min) + 1))
    return [_tile_name(product, lat, lon) for lat in lats for lon in lons]


def _download_file(url: str, local_path: Path) -> None:
    """Download url to local_path, writing it atomically."""
    import requests

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_name(local_path.name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp_path.replace(local_path)


def _read_tile_list(static_dir: Path, product: DemProduct) -> set[str]:
    tile_list_path = get_dem_dir(static_dir, product) / DEM_TILE_LIST_FILE_NAME
    return set(tile_list_path.read_text().split())


def get_dem_tile_paths(
        static_dir: Path,
        product: DemProduct,
        extent: tuple[float, float, float, float],
) -> list[Path]:
    """
    Paths of the local DEM tiles that cover the given extent. Tiles that do
    not exist in the product (because they would only contain sea) are
    skipped.

    Raises FileNotFoundError if the tile list or some of the tiles have not
    been downloaded.
    """
    dem_dir = get_dem_dir(static_dir, product)
    tile_list_path = dem_dir / DEM_TILE_LIST_FILE_NAME
    if not tile_list_path.exists():
        raise FileNotFoundError(tile_list_path)
    existing_tiles = _read_tile_list(static_dir, product)

    paths = [
        dem_dir / f"{name}.tif"
        for name in _tile_names_for_extent(product, extent)
        if name in existing_tiles
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(*missing)
    return paths


def get_cartopy_dir(static_dir: Path) -> Path:
    """Directory used as cartopy's data_dir inside the static directory."""
    return Path(static_dir) / CARTOPY_SUBDIR


def configure_cartopy(static_dir: Path) -> None:
    """
    Make cartopy read (and, if they are missing, download) its shapefiles
    from the static directory instead of its default location.
    """
    import cartopy

    cartopy.config["data_dir"] = get_cartopy_dir(static_dir)


def _gshhs_path() -> Path:
    import cartopy
    from cartopy.io import Downloader

    downloader = Downloader.from_config(
        ("shapefiles", "gshhs", GSHHS_SCALE, GSHHS_LEVEL)
    )
    format_dict = {
        "config": cartopy.config, "scale": GSHHS_SCALE, "level": GSHHS_LEVEL
    }
    return Path(downloader.target_path(format_dict))


def _natural_earth_coastline_path(resolution: str) -> Path:
    import cartopy
    from cartopy.io import Downloader

    downloader = Downloader.from_config(
        ("shapefiles", "natural_earth", resolution, "physical", "coastline")
    )
    format_dict = {
        "config": cartopy.config, "category": "physical",
        "name": "coastline", "resolution": resolution,
    }
    return Path(downloader.target_path(format_dict))


def download_dem(
        static_dir: Path, domains: Iterable[Domain], show_progress: bool = True
) -> None:
    """
    Download, inside the static directory, the Copernicus DEM tiles that
    cover the given domains (skipping the ones already downloaded).
    """
    from tqdm import tqdm

    to_download: dict[Path, str] = {}
    for domain in domains:
        product = get_dem_product(domain)
        dem_dir = get_dem_dir(static_dir, product)

        tile_list_path = dem_dir / DEM_TILE_LIST_FILE_NAME
        if not tile_list_path.exists():
            _download_file(
                f"{product.bucket_url}/{DEM_TILE_LIST_FILE_NAME}",
                tile_list_path,
            )
        existing_tiles = _read_tile_list(static_dir, product)

        for name in _tile_names_for_extent(product, domain_extent(domain)):
            local_path = dem_dir / f"{name}.tif"
            if name in existing_tiles and not local_path.exists():
                to_download[local_path] = f"{product.bucket_url}/{name}/{name}.tif"

    for local_path, url in tqdm(
        to_download.items(), desc="Downloading DEM tiles",
        disable=not show_progress,
    ):
        _download_file(url, local_path)


def download_cartography(static_dir: Path) -> None:
    """
    Download the coastline shapefiles used by the plot (GSHHS and Natural
    Earth) inside the static directory, unless they already exist.
    """
    from cartopy.io import shapereader

    configure_cartopy(static_dir)
    get_cartopy_dir(static_dir).mkdir(parents=True, exist_ok=True)

    # These calls download the files only if they are not already in
    # cartopy's data_dir
    shapereader.gshhs(scale=GSHHS_SCALE, level=GSHHS_LEVEL)
    for resolution in NATURAL_EARTH_COASTLINE_RESOLUTIONS:
        shapereader.natural_earth(
            resolution=resolution, category="physical", name="coastline"
        )


def download_static_data(
        static_dir: Path, domains: Iterable[Domain], show_progress: bool = True
) -> None:
    """
    Download all the static data needed to plot the given domains inside
    the static directory.
    """
    static_dir = Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    download_dem(static_dir, domains, show_progress=show_progress)
    download_cartography(static_dir)


def check_static_data(
        static_dir: Path, domain: Domain, need_dem: bool, coastline: str
) -> None:
    """
    Check that the static data needed by a plot is available inside the
    static directory, so that the plot never tries to download anything
    (which would fail on nodes without internet access).

    Raises FileNotFoundError, listing the missing files, otherwise.
    """
    configure_cartopy(static_dir)

    required = []
    if need_dem:
        product = get_dem_product(domain)
        try:
            required.extend(
                get_dem_tile_paths(static_dir, product, domain_extent(domain))
            )
        except FileNotFoundError as e:
            required.extend(Path(p) for p in e.args)
    if coastline == "GSHHFS":
        required.append(_gshhs_path())
    elif coastline == "cartopy10m":
        required.append(_natural_earth_coastline_path("10m"))
    elif coastline == "cartopy50m":
        required.append(_natural_earth_coastline_path("50m"))

    missing = [p for p in required if not p.exists()]
    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Some static data is missing from {static_dir}:\n"
            f"{missing_list}\n"
            f"Run `comertine download-static -l <static_dir> -d {domain.name}` "
            "on a machine with internet access first."
        )
