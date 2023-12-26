# import os
import sys
import datetime

sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")


from tudatpy.data.mpc import BatchMPC

from tudatpy.kernel.numerical_simulation import environment_setup

# List the bodies for our environment
bodies_to_create = [
    "Sun",
    # "Earth",
]
global_frame_origin = "Sun"
global_frame_orientation = "J2000"
# Create system of bodies
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)
