from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel


class Domain(BaseModel):
    name: str
    long_name: str
    resolution: Literal["regional", "vhr"]


NAD = Domain(name="NAD", long_name="Northern Adriatic", resolution="regional")
SAD = Domain(name="SAD", long_name="Southern Adriatic", resolution="regional")
ION = Domain(name="ION", long_name="Ionian Sea", resolution="regional")
TYR = Domain(name="TYR", long_name="Tyrrhenian Sea", resolution="regional")
LIG = Domain(name="LIG", long_name="Ligurian Sea", resolution="regional")
SAR = Domain(name="SAR", long_name='Sea of Sardinia', resolution="regional")
SIC = Domain(name="SIC", long_name="Sicily", resolution="regional")

HGT = Domain(name="HGT", long_name="Gulf of Trieste", resolution="vhr")


DOMAINS = MappingProxyType(
    {d.name: d for d in [NAD, SAD, ION, TYR, LIG, SAR, SIC, HGT]}
)
