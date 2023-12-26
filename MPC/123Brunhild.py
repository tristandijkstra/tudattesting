# import os
import sys

sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")
import pandas as pd
import numpy as np

# import astroquery.mpc.MPC as astroqueryMPC
from astroquery.mpc import MPC
import datetime
from typing import Union
import time

from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion

# from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u

from pac.data import BatchMPC
spice.load_standard_kernels()
# spice.load_kernel(r"codes_300ast_20100725.bsp")

central_bodies = ["Sun"]
bodies_to_create = ["Sun", "Earth", "Mars", "Jupiter", "Saturn"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# asteroid 123 Brunhild
mpcCodes = [123]

# get observations from astroquery
# NOTE
start = time.perf_counter()
batch = BatchMPC()
batch.get_observations(mpcCodes)
end = time.perf_counter()
print(f"get_observations() runtime = {end-start}")

# filter
# Maybe it should be filter out stations
# batch.filter(stations=["T05"], epoch_start=86400 * 23 * 365)
# batch.filter(stations=["T05", "W68", "W92", "W96"], epoch_start=86400 * 23 * 365)
# batch.filter(epoch_start=86400 * 23 * 365.25, stations_exclude=["C51", "C57"])
# batch.filter(epoch_start=datetime.datetime(2023, 1, 1))
batch.filter(epoch_start=datetime.datetime(2022, 1, 1), stations_exclude=["C51", "C57"])

print(f"BATCH SIZE = {batch.size}")

# process into tudat
start = time.perf_counter()
# NOTE
observation_collection, links = batch.to_tudat(bodies=bodies)
end = time.perf_counter()
print(f"to_tudat() runtime = {end-start}")


epoch_start = batch.epoch_start - (86400 * 10)
epoch_end = batch.epoch_end

start = time.perf_counter()
# create settings
# NOTE
observation_settings_list = list()
for link in links:
    observation_settings_list.append(observation.angular_position(link))

# acceleration stuff
generic_acceleration_set = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
)
# NOTE
acceleration_settings = {}
for body in batch.MPCcodes:
    acceleration_settings[str(body)] = generic_acceleration_set

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, batch.MPCcodes, central_bodies
)

# Set the initial state of the asteroid, approximate, wikipedia
MUsun = bodies.get("Sun").gravitational_parameter
# initial_state = element_conversion.keplerian_to_cartesian_elementwise(
#     gravitational_parameter=MUsun,
#     semi_major_axis=403e9,
#     eccentricity=0.12,
#     inclination=np.deg2rad(6),
#     argument_of_periapsis=np.deg2rad(125),
#     longitude_of_ascending_node=np.deg2rad(307),
#     true_anomaly=np.deg2rad(317),  # random number
# )

initial_state = [2.313864909208807E+08, 1.336874567469971E+08, 9.965771104798709E+07, -2.093187173311458E+01, -6.397890048044714E+00, -2.227628603462231E+00]
initial_state = np.array(initial_state) * 1000
# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(epoch_start)


dt = 6000
# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    epoch_end, -dt, propagation_setup.integrator.rkf_78, -dt, -dt, 1.0, 1.0
)

# integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
#     60, propagation_setup.integrator.CoefficientSets.rk_4
# )
# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies=central_bodies,
    acceleration_models=acceleration_models,
    bodies_to_integrate=batch.MPCcodes,
    initial_states=initial_state,
    initial_time=epoch_end,
    integrator_settings=integrator_settings,
    termination_settings=termination_condition,
)

# Setup parameters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(
    propagator_settings, bodies
)

# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(
    parameter_settings, bodies, propagator_settings
)

end = time.perf_counter()
print(f"general setup runtime = {end-start}")
# #################################################################
# #################################################################
# #################################################################
# #################################################################


start = time.perf_counter()
estimator = numerical_simulation.Estimator(
    bodies=bodies,  # correct
    estimated_parameters=parameters_to_estimate,  # correct
    observation_settings=observation_settings_list,  # correct
    # integrator_settings = integrator_settings, # missing
    propagator_settings=propagator_settings,  #
    integrate_on_creation=True,  # makes things really slow,
)

end = time.perf_counter()
print(f"estimator class runtime = {end-start}")

start = time.perf_counter()
# print(observation_set_list)
pod_input = estimation.EstimationInput(
    observations_and_times=observation_collection,
    convergence_checker=estimation.estimation_convergence_checker(
        maximum_iterations=10,
    )
    # parameters_to_estimate.parameter_set_size
)

pod_input.define_estimation_settings(reintegrate_variational_equations=True)
end = time.perf_counter()
print(f"pod input runtime = {end-start}")
print("running estimation")
pod_output = estimator.perform_estimation(pod_input)


# ##################################################
# ##################################################
# ##################################################
# RESULTS
print("final output:")
print(pod_output)
print(parameters_to_estimate.parameter_vector)
print("Kepler:")
keplerElements = element_conversion.cartesian_to_keplerian(
    parameters_to_estimate.parameter_vector, MUsun
)

print(keplerElements)
print(f"a = {keplerElements[0]} m")
print(f"e = {keplerElements[1]}")
print(f"i = {np.degrees(keplerElements[2])} deg")
print(f"omega = {np.degrees(keplerElements[3])} deg")
print(f"RAAN = {np.degrees(keplerElements[4])} deg")
print(f"theta = {np.degrees(keplerElements[5])} deg")

print("ERRORS:")
print(
    f"a = {(keplerElements[0] - (2.6962986 * 149.6e9))/149.6e9} AU | {(keplerElements[0] - (2.6962986 * 149.6e9))/1000} km"
)
print(f"e = {keplerElements[1]-0.1210537}")
print(f"i = {np.degrees(keplerElements[2]) - 6.41021} deg")
print(f"omega = {np.degrees(keplerElements[3])-124.96775} deg")
# print(f"RAAN = {np.degrees(keplerElements[4])} deg")
# print(f"theta = {np.degrees(keplerElements[5])} deg")


# COMPARISON
# https://www.minorplanetcenter.net/db_search/show_object?utf8=%E2%9C%93&object_id=123
