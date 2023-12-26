import sys
sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")

from tudatpy.data import BatchMPC
# from pac.data import BatchMPC
from tudatpy.kernel.numerical_simulation import environment_setup
import matplotlib.pyplot as plt
import datetime
import numpy as np

# #############################
central_bodies = ["Sun"]
bodies_to_create = ["Sun", "Earth", "Mars", "Jupiter", "Saturn"]
global_frame_origin = "Sun"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)
bodies = environment_setup.create_system_of_bodies(body_settings)
# #############################

# BASIC USAGE
# 1 Ceres
asteroidMPCcodes = [123]

# # 16 Psyche
# asteroidMPCcodes = [16]

batch = BatchMPC()

batch.get_observations(asteroidMPCcodes)

batch.filter(
    bands=None,
    observatories=None,
    # observatories_exclude=,
    epoch_start=datetime.datetime(2000, 1, 1),
    # epoch_start=None,
    epoch_end=None,
)


fig = batch.plotObservations()
fig.savefig("hii.pdf")
# plt.show()

batch.summariseBatch()

fig, ax = plt.subplots(2,1, figsize=(9,6))
ax[0].plot(batch.table.epochUTC, np.degrees(batch.table.RA))
ax[1].plot(batch.table.epochUTC, np.degrees(batch.table.DEC))
fig.savefig("time.pdf")

observation_set_list, links = batch.to_tudat(bodies=bodies)


print(batch.table)
# potential future usage: one liner ("requires some thinking")
# notice the lack of brackets in the class creation
# obsCollection, links = BatchMPC.get_observations(asteroidMPCcodes).filter(stations_out=["C51", "C57"],).to_tudat(bodies=bodies)

# from existing pandas or astroquery table
# batch = BatchMPC.from_astroquery()
# batch = BatchMPC.from_pandas()