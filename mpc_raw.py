import sys

sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")
from tudatpy.data.mpc import BatchMPC
from tudatpy.data.horizons import HorizonsQuery

from tudatpy.numerical_simulation import environment_setup
from tudatpy.interface import spice

import numpy as np
import datetime
import matplotlib.pyplot as plt

from astroquery.mpc import MPC


# obs = MPC.get_observations(433, get_raw_response=True, get_mpcformat=False)
obs = MPC.get_observations(433)
print(obs)
print(obs.columns)
print(obs.to_pandas().iloc[0]["catalog"])
print(type(obs.to_pandas().iloc[0]["catalog"]))
print(obs.to_pandas().iloc[-1]["catalog"])
print(type(obs.to_pandas().iloc[-1]["catalog"]))
# tst = "00433        KB2023 08 22.91881 21 01 09.72 -08 20 25.0          11.4 GXEQ114K19"
# print(len(tst))
# print(tst[72])
# print(tst[72 + 1])
