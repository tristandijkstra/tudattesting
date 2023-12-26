# import os
import sys

sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")
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

# from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u

# from pac.data import BatchMPC
from tudatpy.data.mpc import BatchMPC
# import tudatpy.data as data

import matplotlib.pyplot as plt

from acc import bods, allAccels

from astroquery.mast import MastMissions, MastMissionsClass
# SPICE KERNELS
spice.load_standard_kernels()
spice.load_kernel(r"codes_300ast_20100725.bsp")
spice.load_kernel(r"codes_300ast_20100725.tf")

# TESS - https://archive.stsci.edu/missions/tess/models/
# https://archive.stsci.edu/tess/bulk_downloads.html
# https://archive.stsci.edu/missions-and-data/tess/data-products.html#mod_eng

# 229 = Aug 17
# spice.load_kernel(r"TESS_EPH_DEF_2023229_21.bsp")
spice.load_kernel(r"TESS_EPH_PRE_2021151_21.bsp")
# spice.load_kernel(r"de430.bsp")
# spice.load_kernel(r"earth_070425_370426_predict.bpc")
codes = [433, 1]

batch = BatchMPC()
batch.get_observations(codes)
batch.filter(
    epoch_start=datetime.datetime(2021, 6, 1),
    epoch_end=datetime.datetime(2021, 6, 30),
    observatories_exclude=["C51", "C59"],
)

batch.summary()

# spice.get_body_properties("TESS", "bodynm", 1)
bodies_to_create = bods 
bodies_to_create = bods
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

bodies.add_body(Body())

initial_state_TESS = spice.get_body_cartesian_state_at_epoch("-95", "Earth", "J2000", "NONE", batch.epoch_start)

print(initial_state_TESS)