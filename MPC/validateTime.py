import sys
sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")

from pac.data import BatchMPC
from tudatpy.kernel.numerical_simulation import environment_setup


# BASIC USAGE
asteroidMPCcodes = [123]


batch = BatchMPC()

batch.get_observations(asteroidMPCcodes)

# print("hello")
# print(batch.table)
# print("hello")
# print(batch.table)
# print("hello")
batch.filter(
    # bands=None,
    stations=["T05"],
    # stations_out=["C51", "C57"],
    # epoch_start=None,
    # epoch_end=None,
)

# print(batch.table)
print(batch.table.iloc[-1].loc[["epoch", "epochJ2000secondsTDB", "epochUTC"]])