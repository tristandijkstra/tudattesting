import sys
import datetime
import matplotlib.pyplot as plt
sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")
from tudatpy.data.horizons import HorizonsQuery
import time
import matplotlib.pyplot as plt
import numpy as np

start = time.perf_counter()
query = HorizonsQuery(
    query_id="-151",
    location="@SSB",
    epoch_start=datetime.datetime(2018, 10, 21, 0, 0),
    # epoch_end=((23*365))*86400,
    epoch_end=datetime.datetime(2023, 9, 1, 4, 21),
    # epoch_end=datetime.datetime(2022, 12, 26, 12, 10),
    epoch_step="1d",
    extended_query=True
)
# query = HorizonsQuery(
#     query_id="-",
#     location="@SSB",
#     epoch_list=np.array([0, 3]),
#     extended_query=True
# )

print(query.vectors())
print(query.name)
print(query.designation)
print(query._object_type)
# print(query._format_time_range(((23*365))*86400))
# query = HorizonsQuery(
#     query_id="4;",
#     location="@SSB",
#     # epoch_list=list(np.arange(0, 1e7, int(1e6))+6e8),
#     epoch_list=list(np.linspace(6e8, 6e8+1e6, 200)),
#     extended_query=True
# )

# a = query._format_time(datetime.datetime(2023, 9, 1, 4, 21, 1, 1))
# a = query._format_time(1e7)
# print(a)
# print(len(a))
# raise ValueError("h")

# print("RUN OUTPUT")
# print(sum(query.query_lengths))
# vec = query.carthesian(as_dataframe=True)
# vec['diff_time'] = vec['epochJ2000secondsTDB'].diff()
# end = time.perf_counter()
# print(f"duration = {round(end-start, 3)} seconds")

# print(vec)

# print((vec.diff_time.min()-(20*60))*1e3)
# print((vec.diff_time.max()-(20*60))*1e3)
# print(vec.diff_time.min())
# print(vec.diff_time.max())
# print(vec.diff_time.mean())
# print(vec.diff_time.median())

# print(len(vec))
# # print(len(vec.drop_duplicates("datetime_str")))
# print("HELLO")
# print(print(len(query.queries[0].uri)))
# print("HELLO")