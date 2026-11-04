import argparse
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from comertine.tools.time_conversions import parse_timedelta

def _parse_date_arg(value: str) -> np.datetime64:
    """Parse a "YYYY-MM-DD" or "YYYYMMDD" CLI argument into an
    np.datetime64[s]."""
    for fmt in ("%Y-%m-%d",):
        try:
            return np.datetime64(datetime.strptime(value, fmt), "s")
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"invalid date {value!r}: expected YYYY-MM-DD or YYYYMMDD"
    )


def _parse_timedelta_arg(value: str) -> np.timedelta64:
    """Parse a duration like "120s", "30m", "8h" or "1d" into a
    np.timedelta64."""
    try:
        return parse_timedelta(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _parse_speed_factor_arg(value: str) -> float | str:
    """Parse a positive float, or the string "auto"."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        factor = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid speed factor {value!r}: expected a positive number "
            "or 'auto'"
        ) from None
    if not factor > 0:
        raise argparse.ArgumentTypeError(
            f"invalid speed factor {value!r}: it must be positive"
        )
    return factor


def _parse_existing_dir(value: str) -> Path:
    """Parse a CLI argument into an existing directory."""
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory {path} does not exist")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} is not a directory")
    return path


def read_cli_args(domain_names: Iterable[str]):
    domain_names = tuple(domain_names)

    parser = argparse.ArgumentParser(prog="comertine")
    subparsers = parser.add_subparsers(dest="action", required=True)

    progress_parent = argparse.ArgumentParser(add_help=False)
    progress_parent.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="Show a progress bar (default: enabled).",
    )

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument(
        "-d", "--domain", choices=domain_names, required=True
    )
    download_parser.add_argument(
        "-t", "--date", type=_parse_date_arg, required=False, default=None,
        help="Date to compute, as YYYY-MM-DD (default: today).",
    )
    download_parser.add_argument(
        "-l", "--local-path", type=_parse_existing_dir, required=True,
        help="Path to the local directory where the data of the model will be "
             "stored.",
    )

    download_static_parser = subparsers.add_parser(
        "download-static", parents=[progress_parent],
        help="Download the static data (land elevation and coastlines) "
             "needed by the plot. Run it on a machine with internet access.",
    )
    download_static_parser.add_argument(
        "-d", "--domain", choices=domain_names, nargs="+", default=None,
        help="Domains whose land elevation will be downloaded (default: "
             "all of them).",
    )
    download_static_parser.add_argument(
        "-l", "--local-path", type=Path, required=True,
        help="Path to the local directory where the static data will be "
             "stored (created if it does not exist). Pass the same "
             "directory to the --static-dir option of the plot action.",
    )

    compute_parser = subparsers.add_parser("compute", parents=[progress_parent])
    compute_parser.add_argument(
        "-d", "--domain", choices=domain_names, required=True
    )
    compute_parser.add_argument(
        "-t", "--date", type=_parse_date_arg, required=False, default=None,
        help="Date to compute, as YYYY-MM-DD (default: today).",
    )
    compute_parser.add_argument(
        "-l", "--local-path", type=_parse_existing_dir, required=True,
        help="Path to the local directory where the data of the model is "
             "stored.",
    )
    compute_parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="The directory where the trajectories of the particles will be "
             "stored. If a flag like ${DOMAIN} or ${DATE} is inside the path, "
             "it will be replaced with the correct value",
    )

    # The following options are used to build a ParticleConfig: the ones
    # left to None are not passed to it, so they get its default value
    config_group = compute_parser.add_argument_group("simulation parameters")
    config_group.add_argument(
        "--time-interval", type=_parse_timedelta_arg, default=None,
        help="Time step of the simulation, e.g. 120s (default: 120s).",
    )
    config_group.add_argument(
        "--batch-time-length", type=_parse_timedelta_arg, default=None,
        help="For how long each batch of particles is followed, e.g. 8h "
             "(default: 8h).",
    )
    config_group.add_argument(
        "--batch-every", type=_parse_timedelta_arg, default=None,
        help="How often a new batch of particles is spawned, e.g. 30m "
             "(default: 30m).",
    )
    config_group.add_argument(
        "--batch-size", type=int, default=None,
        help="Number of particles of each batch (default: 500).",
    )
    config_group.add_argument(
        "--increased-speed-factor", type=_parse_speed_factor_arg,
        default="auto",
        help="Factor that multiplies the velocity field moving the "
             "particles, so that their motion is more visible: a positive "
             "number, or 'auto' to choose it from the resolution of the "
             "domain (7.5 for regional domains, 1.5 for vhr ones). "
             "Default: auto.",
    )
    config_group.add_argument(
        "--seed", type=int, default=None,
        help="Seed of the random generator that spawns the particles "
             "(default: 0).",
    )

    cluster_group = compute_parser.add_argument_group("dask cluster")
    cluster_group.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of worker processes of the dask LocalCluster "
             "(default: one per CPU).",
    )
    cluster_group.add_argument(
        "--memory-limit", type=str, default=None,
        help="Memory limit of each dask worker, e.g. 4GiB (default: chosen "
             "by dask).",
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
        "--background-variable", choices=["speed", "thetao", "so"],
        default="speed",
        help="Which field to draw as the background color field: current "
             "speed (default), temperature (thetao) or salinity (so).",
    )
    plot_parser.add_argument(
        "--background-cmap", type=str, default="viridis",
        help="Colormap used to draw the background field: a matplotlib "
             "colormap name (default: viridis), or a cmocean colormap "
             "name prefixed with 'cmo:' (e.g. cmo:thermal).",
    )
    plot_parser.add_argument(
        "--vmin", type=float, default=None,
        help="Lower bound of the background field's colorbar (default: "
             "the dataset's 5th percentile of that field's values).",
    )
    plot_parser.add_argument(
        "--vmax", type=float, default=None,
        help="Upper bound of the background field's colorbar (default: "
             "the dataset's 95th percentile of that field's values).",
    )
    plot_parser.add_argument(
        "--tight-layout", action="store_true", default=False,
        help="Recompute the width of the frames from the aspect ratio of "
             "the map, so that the image is only slightly wider than the "
             "map itself (leaving room for the axis labels and the "
             "colorbar) instead of using a fixed 16:9 canvas.",
    )
    plot_parser.add_argument(
        "--dark-mode", action="store_true", default=False,
        help="Draw the frames using matplotlib's 'dark_background' style.",
    )
    plot_parser.add_argument(
        "-l", "--local-path", type=_parse_existing_dir, required=True,
        help="Path to the local directory where the data of the model is "
             "stored (the same one passed to the download and compute "
             "actions).",
    )
    plot_parser.add_argument(
        "-s", "--static-dir", type=_parse_existing_dir, default=None,
        help="The directory where the static data has been stored by the "
             "download-static action. Required unless both --land-filler "
             "and --coastline are 'none'.",
    )
    plot_parser.add_argument(
        "-i", "--input-dir", type=Path, required=True,
        help="The directory where the trajectories of the particles are "
             "stored.",
    )
    plot_parser.add_argument(
        "-o", "--output-dir", type=str, default="/dev/shm/particles_plot",
        help="The directory where the frame PNGs will be stored (default: "
             "/dev/shm/particles_plot). If a flag like ${DOMAIN} or "
             "${DATE} is inside the path, it will be replaced with the "
             "correct value.",
    )

    args = parser.parse_args()

    if args.action == "plot" and args.static_dir is None and (
        args.land_filler != "none" or args.coastline != "none"
    ):
        parser.error(
            "--static-dir is required unless both --land-filler and "
            "--coastline are 'none'"
        )

    return args
