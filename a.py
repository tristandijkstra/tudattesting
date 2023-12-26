import sys
import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")

# Tudat imports for propagation and estimation
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation, constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# import MPC interface
from tudatpy.data.mpc import BatchMPC
from tudatpy.data.horizons import HorizonsQuery, HorizonsBatch

# other useful modules
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from astropy.time import Time


from tudatpy.numerical_simulation.environment_setup.ephemeris import JPL_HorizonsEphemerisSettings

