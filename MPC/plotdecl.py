import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pac.data import BatchMPC



# get observations from astroquery
mpcCodes = [123]
batch = BatchMPC()
batch.get_observations(mpcCodes)

# filter
batch.filter(stations=["T05"], epoch_start=86400*20*365)

print(f"BATCH SIZE = {batch.size}")
# process into tudat
# observation_set_list, links = batch.to_tudat(bodies=bodies)

fig, ax = plt.subplots(2, 1)

ax[0].plot(batch.table.epoch, np.degrees(batch.table.RA))
ax[1].plot(batch.table.epoch, np.degrees(batch.table.DEC))
ax[0].set_title("RA")
ax[1].set_title("DEC")

# plt.show()
fig.savefig("/mnt/c/Users/Trez/Desktop/astroquerytest/hi.png")