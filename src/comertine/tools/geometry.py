from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Extent:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def central_longitude(self) -> float:
        return (self.lon_min + self.lon_max) / 2.

    def central_latitude(self) -> float:
        return (self.lat_min + self.lat_max) / 2.


@dataclass(frozen=True, slots=True)
class Grid:
    latitude: NDArray[np.float64]
    longitude: NDArray[np.float64]

    def __post_init__(self):
        if not isinstance(self.latitude, np.ndarray):
            raise TypeError("latitude must be a numpy array")

        if not isinstance(self.longitude, np.ndarray):
            raise TypeError("longitude must be a numpy array")

    def latitude_boundaries_and_sizes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return get_borders_and_sizes(self.latitude)

    def longitude_boundaries_and_sizes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return get_borders_and_sizes(self.longitude)


def get_borders_and_sizes(v: np.typing.NDArray):
    """
    Compute borders from the centers

    Given a 1D array that contains the position of the centers of some cells,
    it computes the position of the boundaries between the two cells supposing
    that the boundaries are exactly in the middle between the two centers.

    Args:
        v: the array with the position of the centers

    Returns:
        A tuple with two elements:
        - A new array with the position of the boundaries
        - A new array with the length of each cell (i.e., the distance between
          two consecutive boundaries)
    """
    v_borders = np.empty(v.shape[0] + 1, dtype=v.dtype)
    v_borders[1:-1] = (v[:-1] + v[1:]) / 2.
    v_borders[0] = v[0] - (v[1] - v[0]) / 2.
    v_borders[-1] = v[-1] + (v[-1] - v[-2]) / 2.
    return v_borders, v_borders[1:] - v_borders[:-1]


def get_particles_cells(particles: xr.Dataset, grid: Grid):
    """
    Compute the indices where the particles are

    Args:
        particles: a Dataset with the position of the particles
        grid: the grid of the model

    """
    particle_lat = particles["latitude"].values
    particle_lon = particles["longitude"].values

    latitude_borders, _ = grid.latitude_boundaries_and_sizes()
    longitude_borders, _ = grid.longitude_boundaries_and_sizes()

    lat_index = np.searchsorted(latitude_borders[:-1], particle_lat) - 1
    lon_index = np.searchsorted(longitude_borders[:-1], particle_lon) - 1

    particles["latitude_index"] = xr.DataArray(lat_index, dims=("particle",))
    particles["longitude_index"] = xr.DataArray(lon_index, dims=("particle",))


def compute_new_coords(current_lat:np.ndarray, current_lon:np.ndarray, x_velocity:np.ndarray, y_velocity:np.ndarray, dt_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the new coordinates of the particles.

    Advects the current coordinates by the given velocity field over a
    time step of `dt_seconds`, then converts the resulting displacement from
    meters to degrees.

    Args:
        current_lat: The current latitude of the particle (in degrees).
        current_lon: The current longitude of the particle (in degrees).
        x_velocity: The velocity in the x direction (in m/s).
        y_velocity: The velocity in the y direction (in m/s).
        dt_seconds: The time step in seconds.
    """
    # Earth radius in meters
    earth_radius = 6371000.0

    y_movement = y_velocity * dt_seconds
    x_movement = x_velocity * dt_seconds

    # Convert y_movement (north-south) from meters to degrees
    lat_change = np.degrees(y_movement / earth_radius)

    # Convert x_movement (east-west) from meters to degrees
    # Account for convergence of meridians at different latitudes
    lon_change = np.degrees(
        x_movement / (earth_radius * np.cos(np.radians(current_lat))))

    new_lat = current_lat + lat_change
    new_lon = current_lon + lon_change

    return new_lat, new_lon
