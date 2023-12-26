import sys
import datetime
import matplotlib.pyplot as plt
sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")
from tudatpy.data.horizons import HorizonsQuery
import time
import matplotlib.pyplot as plt
import numpy as np
from astroquery.jplhorizons import Horizons


# print(Horizons(id='6000;', id_type=None).ephemerides())  
query = HorizonsQuery(
    query_id="65121;",
    location="@SSB",
    epoch_list=[22*365*86400],
)
vec = query.carthesian()

print(query._target_full_name)
print(query.name)
print(query.designation)
print(query.MPC_number)