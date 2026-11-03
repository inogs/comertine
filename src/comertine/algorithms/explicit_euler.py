import numpy as np
import xarray as xr

from comertine.algorithms.interpolations import bilinear_interpolation
from comertine.tools.arrays import enlarge_array
from comertine.tools.geometry import Grid
from comertine.tools.geometry import compute_new_coords
from comertine.tools.geometry import get_particles_cells


def evolve_particles(
    particles: xr.Dataset,
    uo: np.ndarray,
    vo: np.ndarray,
    grid: Grid,
    dt_seconds: float,
) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """
    Advance the particles by one explicit Euler step.

    The velocity is bilinearly interpolated on the position of each particle
    and then used to move the particle for `dt_seconds`. Particles that have
    already left the domain (outside the grid or on land) do not move.

    Args:
        particles: The current state of the particles (as produced by
            `initialize_particles` or by a previous call of this function).
        uo: The eastward velocity on the grid (NaN on land).
        vo: The northward velocity on the grid (NaN on land).
        grid: The grid of the model.
        dt_seconds: The length of the time step, in seconds.

    Returns:
        A tuple with the new state of the particles and the velocities
        (uo and vo) interpolated on the particles before the step (NaN for
        the particles that are not active).
    """
    uo_enlarged = enlarge_array(np.where(np.isfinite(uo), uo, 0))
    vo_enlarged = enlarge_array(np.where(np.isfinite(vo), vo, 0))

    latitude_borders, _ = grid.latitude_boundaries_and_sizes()
    longitude_borders, _ = grid.longitude_boundaries_and_sizes()

    grid_lat_min = np.min(latitude_borders)
    grid_lat_max = np.max(latitude_borders)

    grid_lon_min = np.min(longitude_borders)
    grid_lon_max = np.max(longitude_borders)

    lat_enlarged = enlarge_array(grid.latitude, fill_value=grid_lat_min - 1)
    lon_enlarged = enlarge_array(grid.longitude, fill_value=grid_lon_min - 1)
    lat_enlarged[-1] = grid_lat_max + 1
    lon_enlarged[-1] = grid_lon_max + 1

    latitude = particles.latitude.values
    longitude = particles.longitude.values
    inside_domain = particles.inside_domain.values
    latitude_index = particles.latitude_index.values
    longitude_index = particles.longitude_index.values

    # Particles that have already left the domain (grid bounds or land)
    # stay where they are: only the active ones are evolved.
    active = inside_domain

    active_latitude = latitude[active]
    active_longitude = longitude[active]
    active_latitude_index = latitude_index[active]
    active_longitude_index = longitude_index[active]

    reference_center_lat = grid.latitude[active_latitude_index]
    reference_center_lon = grid.longitude[active_longitude_index]

    lat_min_index = np.where(
        active_latitude < reference_center_lat,
        active_latitude_index,
        active_latitude_index + 1
    )
    lon_min_index = np.where(
        active_longitude < reference_center_lon,
        active_longitude_index,
        active_longitude_index + 1
    )

    lat_min = lat_enlarged[lat_min_index]
    lon_min = lon_enlarged[lon_min_index]
    lat_max = lat_enlarged[lat_min_index + 1]
    lon_max = lon_enlarged[lon_min_index + 1]

    assert np.all(active_latitude >= lat_min)
    assert np.all(active_latitude <= lat_max)
    assert np.all(active_longitude >= lon_min)
    assert np.all(active_longitude <= lon_max)

    def _interpolate_variable_on_particles(var_data):
        return bilinear_interpolation(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            v_min_min=var_data[lat_min_index, lon_min_index],
            v_max_min=var_data[lat_min_index + 1, lon_min_index],
            v_min_max=var_data[lat_min_index, lon_min_index + 1],
            v_max_max=var_data[lat_min_index + 1, lon_min_index + 1],
            lat_position=active_latitude,
            lon_position=active_longitude,
        )

    uo_particles = _interpolate_variable_on_particles(uo_enlarged)
    vo_particles = _interpolate_variable_on_particles(vo_enlarged)

    new_active_lat, new_active_lon = compute_new_coords(
        active_latitude,
        active_longitude,
        uo_particles,
        vo_particles,
        dt_seconds=dt_seconds,
    )

    inside_lat = np.logical_and(
        new_active_lat >= grid_lat_min,
        new_active_lat <= grid_lat_max,
    )
    inside_lon = np.logical_and(
        new_active_lon >= grid_lon_min,
        new_active_lon <= grid_lon_max,
    )
    still_inside = np.logical_and(inside_lat, inside_lon)

    # A particle that landed on a cell that was originally NaN (land) has
    # left the domain too, even though it is still within the grid bounds.
    new_active_lat_index = np.searchsorted(latitude_borders[:-1], new_active_lat) - 1
    new_active_lon_index = np.searchsorted(longitude_borders[:-1], new_active_lon) - 1
    on_land = ~np.isfinite(uo[new_active_lat_index, new_active_lon_index])
    still_inside = np.logical_and(still_inside, ~on_land)

    new_latitude = latitude.copy()
    new_longitude = longitude.copy()
    new_inside_domain = inside_domain.copy()

    new_latitude[active] = new_active_lat
    new_longitude[active] = new_active_lon
    new_inside_domain[active] = still_inside

    uo_at_particles = np.full(latitude.shape, np.nan)
    vo_at_particles = np.full(latitude.shape, np.nan)
    uo_at_particles[active] = uo_particles
    vo_at_particles[active] = vo_particles

    new_particles = xr.Dataset(
        {
            "latitude": xr.DataArray(new_latitude, dims=("particle",)),
            "longitude": xr.DataArray(new_longitude, dims=("particle",)),
            "inside_domain": xr.DataArray(new_inside_domain, dims=("particle",)),
        }
    )
    get_particles_cells(new_particles, grid=grid)
    return new_particles, uo_at_particles, vo_at_particles
