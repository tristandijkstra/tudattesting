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
import copy

a = BatchMPC()
a.get_observations([1])
b = copy.copy(a)

a.filter(epoch_start=datetime.datetime(2022, 1 ,1))
c = a.filter(epoch_start=datetime.datetime(2023, 1 ,1), in_place=False)
print("A")
a.summary()
print("B")
b.summary()
print("C")
c.summary()