import numpy as np
import xarray as xr

from comertine.tools.geometry import get_particles_cells


def initialize_particles(n_particles, grid, uo, vo, random_generator):
    latitude_borders, lat_sizes = grid.latitude_boundaries_and_sizes()
    longitude_borders, lon_sizes = grid.longitude_boundaries_and_sizes()

    data_mask = np.isfinite(uo)

    cells_lat_indices = np.arange(0, grid.latitude.shape[0])[:, None]
    cells_lon_indices = np.arange(0, grid.longitude.shape[0])[None, :]

    cells_lat_indices, cells_lon_indices = np.broadcast_arrays(cells_lat_indices, cells_lon_indices)

    cells_lat_indices = cells_lat_indices[data_mask]
    cells_lon_indices = cells_lon_indices[data_mask]

    speed = np.sqrt(uo[data_mask]**2 + vo[data_mask]**2)

    prob_indices = np.sqrt(speed).ravel().cumsum()
    samples = random_generator.uniform(0, prob_indices[-1], size=n_particles)

    shift_lat = random_generator.uniform(0, 1, size=n_particles)
    shift_lon = random_generator.uniform(0, 1, size=n_particles)

    data_cells = np.searchsorted(prob_indices, samples)

    samples_lat_cells = cells_lat_indices[data_cells]
    samples_lon_cells = cells_lon_indices[data_cells]

    lat_position = latitude_borders[samples_lat_cells]
    lon_position = longitude_borders[samples_lon_cells]

    lat_position += lat_sizes[samples_lat_cells] * shift_lat
    lon_position += lon_sizes[samples_lon_cells] * shift_lon

    ds = xr.Dataset(
        {
            "latitude": xr.DataArray(lat_position, dims=("particle",)),
            "longitude": xr.DataArray(lon_position, dims=("particle",)),
            "inside_domain": xr.DataArray(np.ones(n_particles, dtype=bool), dims=("particle",)),
        }
    )

    get_particles_cells(ds, grid=grid)
    return ds
