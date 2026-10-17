"""External static pressure for a rooftop supply fan.

ESP, in inches of water column, is the sum of the pressure drops the fan
has to overcome: filter, coil, supply duct, and diffuser.
"""


def external_static_pressure_in_wc(
    filter_drop: float,
    coil_drop: float,
    duct_drop: float,
    diffuser_drop: float,
) -> float:
    return filter_drop + coil_drop + duct_drop + diffuser_drop
