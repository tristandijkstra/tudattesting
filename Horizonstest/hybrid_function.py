# import os
import sys
import datetime

sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")
# from tudatpy.data.mpc import BatchMPC
# from tudatpy.data.horizons import HorizonsQuery

from tudatpy.numerical_simulation.environment_setup.ephemeris import jpl_horizons
from tudatpy.numerical_simulation import environment_setup

from tudatpy.interface import spice
spice.load_standard_kernels()


juice_eph_settings = jpl_horizons(
    horizons_query="433",
    horizons_location="500@SSB",
    frame_origin="Earth",  # tudat frame origin and orientation
    frame_orientation="J2000",
    epoch_start=datetime.datetime(2020, 1, 1),
    epoch_end=datetime.datetime(2023, 1, 1),
    epoch_step="1d",
    extended_query=True,
)

bodies_to_create = [
    "Earth",
]
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

body_settings.add_empty_settings("Eros")

body_settings.get("Eros").ephemeris_settings = juice_eph_settings

