from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from comertine.domains import Domain
from comertine.tools.file_names import concretize_template_string
from comertine.tools.file_names import get_data_file_path


URL = "https://mer-model-bgc.ogs.it/output/${DOMAIN}/${DATE}-MER-MITgcmBFM-${DOMAIN}-fc-${FREQUENCY}-v01.zarr"


def _get_url(domain: Domain, date: np.datetime64):
    return concretize_template_string(template_string=URL, domain=domain, date=date)


def download_data(domain: Domain, date: np.datetime64, local_directory: Path, variables: Sequence[str] | None):
    if variables is None:
        variables = ["thetao", "uo", "vo", "so"]
    else:
        variables = tuple(variables)

    download_url = _get_url(domain, date)

    ds = xr.open_zarr(download_url, chunks={})

    ds = ds.squeeze("time", drop=True)
    ds = ds.swap_dims({"step": "valid_time"})
    ds = ds.drop_vars(["step"])
    ds = ds.rename({"valid_time": "time"})

    ds = ds.isel(depth=0)
    vars_to_keep = ["uo", "vo"]

    if "thetao" in variables:
        vars_to_keep.append("thetao")
    if "so" in variables:
        vars_to_keep.append("so")

    vars_to_drop = [d for d in ds.data_vars if d not in vars_to_keep]
    ds = ds.drop_vars(vars_to_drop)

    velocities_file = get_data_file_path(variable="velocities", local_directory=local_directory, domain=domain, date=date, create_parents=True)
    ds[["uo", "vo"]].to_zarr(velocities_file, mode="w")

    for hourly_var in ["thetao", "so"]:
        if hourly_var in variables:
            current_var_file = get_data_file_path(variable=hourly_var, local_directory=local_directory, domain=domain, date=date, create_parents=True)
            ds[[hourly_var]].to_zarr(current_var_file, mode="w")
