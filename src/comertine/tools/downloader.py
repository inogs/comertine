from collections.abc import Iterable
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import xarray as xr

from comertine.domains import Domain
from comertine.tools.file_names import concretize_template_string
from comertine.tools.file_names import get_data_file_path


URL = "https://mer-model-bgc.ogs.it/output/${DOMAIN}/${DATE}-MER-MITgcmBFM-${DOMAIN}-fc-${FREQUENCY}-v01.zarr"

Frequency = Literal["hourly", "daily"]

# The variables that can be downloaded, and the file of the model (hourly or
# daily) that contains each of them
VARIABLE_FREQUENCIES: MappingProxyType[str, Frequency] = MappingProxyType({
    "uo": "hourly",
    "vo": "hourly",
    "thetao": "hourly",
    "so": "hourly",
    "o2": "hourly",
    "chl": "daily",
    "no3": "daily",
})

AVAILABLE_VARIABLES = tuple(VARIABLE_FREQUENCIES)

# Always downloaded: they are needed to move the particles
REQUIRED_VARIABLES = ("uo", "vo")

# The velocities are stored together, in a single file with this name (see
# get_data_file_path); every other variable has its own file
VELOCITIES_FILE_NAME = "velocities"


def _get_url(domain: Domain, date: np.datetime64, frequency: Frequency) -> str:
    return concretize_template_string(
        template_string=URL, domain=domain, date=date, frequency=frequency
    )


def _resolve_variables(variables: Iterable[str] | None) -> tuple[str, ...]:
    """
    Return the variables to download, in the order of AVAILABLE_VARIABLES.

    None means all the available variables; uo and vo are always added.
    """
    if variables is None:
        return AVAILABLE_VARIABLES

    variables = set(variables)
    unknown = sorted(variables - set(AVAILABLE_VARIABLES))
    if unknown:
        msg = (
            f"Unknown variables: {', '.join(unknown)} (available: "
            f"{', '.join(AVAILABLE_VARIABLES)})"
        )
        raise ValueError(msg)

    variables |= set(REQUIRED_VARIABLES)
    return tuple(v for v in AVAILABLE_VARIABLES if v in variables)


def _open_surface_dataset(
        domain: Domain, date: np.datetime64, frequency: Frequency,
        variables: Iterable[str],
) -> xr.Dataset:
    """
    Lazily open the (remote) file of the model with the given frequency.

    Only `variables` are kept, the "time" (the date of the run) and "step"
    (the forecast step) dimensions are replaced by a single "time" dimension
    with the actual time of each frame, and only the first (surface) layer is
    selected. The processing is the same for the hourly and the daily files.
    """
    ds = xr.open_zarr(_get_url(domain, date, frequency), chunks={})

    # In the files of the model, valid_time (the actual time of each step)
    # is a data variable: it must become a coordinate before selecting the
    # variables, or it would be dropped
    ds = ds.set_coords("valid_time")
    ds = ds[list(variables)]
    ds = ds.squeeze("time", drop=True)
    ds = ds.swap_dims({"step": "valid_time"})
    ds = ds.drop_vars(["step"])
    ds = ds.rename({"valid_time": "time"})

    return ds.isel(depth=0)


def download_data(
        domain: Domain,
        date: np.datetime64,
        local_directory: Path,
        variables: Iterable[str] | None = None,
) -> None:
    """
    Download the surface layer of some variables of the model.

    uo and vo are always downloaded, and stored together in the
    "velocities" file; each other variable is stored in its own file (see
    get_data_file_path). Each variable is read from the hourly or the daily
    file of the model, as specified in VARIABLE_FREQUENCIES.

    Args:
        domain: The domain of the model.
        date: The date of the run of the model.
        local_directory: Where the data are stored.
        variables: The variables to download (see AVAILABLE_VARIABLES);
            None means all of them.
    """
    variables = _resolve_variables(variables)

    for frequency in ("hourly", "daily"):
        frequency_variables = [
            v for v in variables if VARIABLE_FREQUENCIES[v] == frequency
        ]
        if not frequency_variables:
            continue

        ds = _open_surface_dataset(domain, date, frequency, frequency_variables)

        # The velocities first, together; then one file for each of the
        # other variables
        groups: dict[str, list[str]] = {}
        for variable in frequency_variables:
            if variable in REQUIRED_VARIABLES:
                groups.setdefault(VELOCITIES_FILE_NAME, []).append(variable)
            else:
                groups[variable] = [variable]

        for file_name, group_variables in groups.items():
            output_file = get_data_file_path(
                variable=file_name, local_directory=local_directory,
                domain=domain, date=date, create_parents=True,
            )
            ds[group_variables].to_zarr(output_file, mode="w")
