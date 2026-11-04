from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel


class BoundaryBox(BaseModel):
    minimum_longitude: float
    maximum_longitude: float
    minimum_latitude: float
    maximum_latitude: float


class Domain(BaseModel):
    name: str
    long_name: str
    resolution: Literal["regional", "vhr"]
    bounding_box: BoundaryBox

_NAD_BBOX = BoundaryBox(
    minimum_longitude = 12.1041672949297,
    minimum_latitude = 42.750001011549,
    maximum_longitude = 17.937500628263,
    maximum_latitude = 45.875001011549,
)

_SAD_BBOX = BoundaryBox(
    minimum_longitude=13.8541673414735,
    minimum_latitude=39.5000007498547,
    maximum_longitude=20.3125006748068,
    maximum_latitude=43.0416674165214,
)

_ION_BBOX = BoundaryBox(
    minimum_longitude = 15.437500716918,
    minimum_latitude = 35.0416670575305,
    maximum_longitude = 19.0208340502513,
    maximum_latitude = 40.5416670575305,
)

_TYR_BBOX = BoundaryBox(
    minimum_longitude = 11.2291672716577,
    minimum_latitude = 38.9166673695506,
    maximum_longitude = 16.2291672716577,
    maximum_latitude = 42.4583340362173,
)

_LIG_BBOX = BoundaryBox(
    minimum_longitude = 7.31250050082149,
    minimum_latitude = 41.5833342509408,
    maximum_longitude = 12.5208338341548,
    maximum_latitude = 44.5000009176075,
)

_SAR_BBOX = BoundaryBox(
    minimum_longitude = 5.89583379647646,
    minimum_latitude = 37.6250005988772,
    maximum_longitude = 11.5208337964765,
    maximum_latitude = 41.7916672655439,
)

_SIC_BBOX = BoundaryBox(
    minimum_longitude = 11.437500610532,
    minimum_latitude = 35.2500004076391,
    maximum_longitude = 16.2291672771987,
    maximum_latitude = 39.0000004076391,
)

_HGT_BBOX = BoundaryBox(
    minimum_longitude = 13.432292294915,
    minimum_latitude = 45.45833459636,
    maximum_longitude = 13.807292295035,
    maximum_latitude = 45.80208459647,
)


NAD = Domain(name="NAD", long_name="Northern Adriatic", resolution="regional", bounding_box=_NAD_BBOX)
SAD = Domain(name="SAD", long_name="Southern Adriatic", resolution="regional", bounding_box=_SAD_BBOX)
ION = Domain(name="ION", long_name="Ionian Sea", resolution="regional", bounding_box=_ION_BBOX)
TYR = Domain(name="TYR", long_name="Tyrrhenian Sea", resolution="regional", bounding_box=_TYR_BBOX)
LIG = Domain(name="LIG", long_name="Ligurian Sea", resolution="regional", bounding_box=_LIG_BBOX)
SAR = Domain(name="SAR", long_name='Sea of Sardinia', resolution="regional", bounding_box=_SAR_BBOX)
SIC = Domain(name="SIC", long_name="Sicily", resolution="regional", bounding_box=_SIC_BBOX)

HGT = Domain(name="HGT", long_name="Gulf of Trieste", resolution="vhr", bounding_box=_HGT_BBOX)


DOMAINS = MappingProxyType(
    {d.name: d for d in [NAD, SAD, ION, TYR, LIG, SAR, SIC, HGT]}
)
