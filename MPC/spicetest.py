# import os
import sys

sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")
import numpy as np
import datetime

from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u

# from pac.data import BatchMPC
from tudatpy.data.mpc import BatchMPC
# import tudatpy.data as data

import matplotlib.pyplot as plt
spice.load_standard_kernels()
print(spice.get_total_count_of_kernels_loaded())
initial_state_Ceres = spice.get_body_cartesian_state_at_epoch("Ceres", "SSB", "J2000", "NONE", 0)