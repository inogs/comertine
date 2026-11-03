from pathlib import Path

import numpy as np

from comertine.domains import (Domain)
from comertine.tools.time_conversions import numpy_to_datetime


def concretize_template_string(template_string: str, domain: Domain, date: np.datetime64, frequency: str = "hourly"):
    replace_vals = {
        "${DOMAIN}": domain.name.upper(),
        "${DATE}": numpy_to_datetime(date).strftime("%Y%m%d"),
        "${FREQUENCY}": frequency
    }
    for old, new in replace_vals.items():
        template_string = template_string.replace(old, new)
    return template_string


def get_data_file_path(variable: str, local_directory: Path, domain: Domain, date: np.datetime64, *, create_parents: bool = False):
    date_str = numpy_to_datetime(date).strftime("%Y%m%d")

    domain_local_directory = local_directory / domain.name.upper()
    path_current_run = domain_local_directory / date_str

    if create_parents:
        domain_local_directory.mkdir(parents=False, exist_ok=True)
        path_current_run.mkdir(parents=False, exist_ok=True)

    return path_current_run / f"{variable}.zarr"
