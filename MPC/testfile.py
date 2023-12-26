import pandas as pd
import numpy as np

# import astroquery.mpc.MPC as astroqueryMPC
from astroquery.mpc import MPC
import datetime
from typing import Union
import time

from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion

from pac.data import BatchMPC

bodies_to_create = ["Sun", "Earth", "Mars", "Jupiter"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

mpcCodes = [123]

start = time.perf_counter()
batch = BatchMPC()
batch.get_observations(mpcCodes)
end = time.perf_counter()
print(f"runtime = {end-start}")

batch.filter(stations=["T05"])

# print(batch.to_tudat(bodies=bodies))


observation_settings_list = [observation_setup]

# TODO FINISH ESTIMATION SCRIPT