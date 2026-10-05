import argparse
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

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

    compute_parser = subparsers.add_parser("compute", parents=[progress_parent])
    compute_parser.add_argument(
        "-d", "--domain", choices=domain_names, required=True
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
    return args
