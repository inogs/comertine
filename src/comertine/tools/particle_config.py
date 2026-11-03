from types import MappingProxyType
from typing import Annotated

import numpy as np
from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PlainSerializer
from pydantic import WithJsonSchema

from comertine.domains import Domain
from comertine.tools.time_conversions import format_timedelta
from comertine.tools.time_conversions import to_timedelta64


# Default values of `ParticleConfig.increased_speed_factor` for each
# resolution of the domains: real advection over a time step would barely
# move the particles from one frame of the animation to the next.
AUTO_SPEED_FACTORS = MappingProxyType({
    "regional": 7.5,
    "vhr": 1.5,
})


def auto_speed_factor(domain: Domain) -> float:
    """Return the default speed factor for the resolution of `domain`."""
    try:
        return AUTO_SPEED_FACTORS[domain.resolution]
    except KeyError:
        msg = (
            f"No automatic speed factor for resolution "
            f"{domain.resolution!r} (domain {domain.name})"
        )
        raise ValueError(msg) from None


def _check_positive(value: np.timedelta64) -> np.timedelta64:
    if value <= np.timedelta64(0, "s"):
        msg = f"the duration must be positive (received {value})"
        raise ValueError(msg)
    return value


# A positive np.timedelta64 (with unit "s"). It can be built from a
# np.timedelta64, a datetime.timedelta or a string like "120s" or "8h", and
# it is serialized as a string in the latter format.
PositiveTimedelta = Annotated[
    np.timedelta64,
    BeforeValidator(to_timedelta64),
    AfterValidator(_check_positive),
    PlainSerializer(format_timedelta, return_type=str),
    WithJsonSchema({"type": "string", "examples": ["120s", "30m", "8h"]}),
]


class ParticleConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        # So that the defaults are normalized like the other values
        validate_default=True,
    )

    # The temporal distance from on frame to another
    time_interval: PositiveTimedelta = np.timedelta64(120, "s")

    # For how much time do we run a single particle?
    batch_time_length: PositiveTimedelta = np.timedelta64(8, "h")

    # The frequency when we span new particles
    batch_every: PositiveTimedelta = np.timedelta64(30, "m")

    # How many particles to we spawn at the same time?
    batch_size: int = Field(default=500, gt=0)

    # Multiplies the velocity field (uo and vo) that moves the particles, so
    # that their motion is more visible on the animation
    increased_speed_factor: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    # Seed of the random generator used to spawn the particles
    seed: int = Field(default=0, ge=0)

    @property
    def time_interval_seconds(self) -> float:
        return float(self.time_interval / np.timedelta64(1, "s"))
