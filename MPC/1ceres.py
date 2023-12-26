# import os
import sys

sys.path.insert(0, r"/mnt/c/Users/Trez/Desktop/tudat-bundle/tudatpy/")
import pandas as pd
import numpy as np

# import astroquery.mpc.MPC as astroqueryMPC
from astroquery.mpc import MPC  # noqa: E402
import datetime
from typing import Union
import time

from tudatpy import constants# noqa: E402
from tudatpy.interface import spice# noqa: E402
from tudatpy import numerical_simulation# noqa: E402
from tudatpy.numerical_simulation import environment_setup# noqa: E402
from tudatpy.numerical_simulation import propagation_setup# noqa: E402
from tudatpy.numerical_simulation import estimation, estimation_setup# noqa: E402
from tudatpy.numerical_simulation.estimation_setup import observation# noqa: E402
from tudatpy.astro import element_conversion# noqa: E402

# from astropy.time import Time
from astropy.coordinates import EarthLocation# noqa: E402
from astropy import units as u# noqa: E402

# from pac.data import BatchMPC
from tudatpy.data.mpc import BatchMPC# noqa: E402
# import tudatpy.data as data

import matplotlib.pyplot as plt# noqa: E402

from acc import bods, allAccels# noqa: E402


spice.load_standard_kernels()

central_bodies = ["SSB"]
# bodies_to_create = ["Sun", "Earth", "Mars", "Jupiter", "Saturn", "Titan",]
bodies_to_create = bods

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# asteroid 1 Ceres
mpcCodes = [1]
bodySpiceName = "Ceres"
# mpcCodes = [4]
# bodySpiceName = "Vesta"
# mpcCodes = [16]
# bodySpiceName = "Psyche"
# mpcCodes = [7]
# bodySpiceName = "2000007" #IRIS
# mpcCodes = [3]
# bodySpiceName = "2000003" # JUNO
# mpcCodes = [2]
# bodySpiceName = "Pallas"
# mpcCodes = [433]
# bodySpiceName = "Eros"

batch = BatchMPC()
# print("HELLO")
# print(batch._observatory_info.query("Code == '006'").loc[:, ["X", "Y", "Z"]].values)
# print(batch._observatory_info.query("Code == '006'").loc[:, ["X", "Y", "Z"]].values - np.array([4786956.594, 177551.835, 4197636.831]))
# print("HELLO")
batch.get_observations(mpcCodes)

# satelliteObservatories = batch._observatory_info.query("X != X").Code.unique()
batch.filter(epoch_start=datetime.datetime(2015, 1, 1))
exclude = [x for x in batch.observatories if x in satelliteObservatories]
print(exclude)
batch.filter(observatories_exclude=exclude)
print(f"BATCH SIZE = {batch.size}")

# print(batch._observatory_info.query("X != X").Code.unique())
# print(batch.observatories)
# batch.summariseBatch()

# process into tudat
bodies.create_empty_body("1")
observation_collection, links = batch.to_tudat(bodies=bodies)

epoch_start = batch.epoch_start - (86400 * 4)
epoch_end = batch.epoch_end
# epoch_end = (2460110.5 - 2451545.0) * 86400
observation_settings_list = list()
for link in list(links.values()):
    observation_settings_list.append(observation.angular_position(link))

# acceleration stuff
# generic_acceleration_set = dict(
#     Sun=[propagation_setup.acceleration.point_mass_gravity()],
#     Mars=[propagation_setup.acceleration.point_mass_gravity()],
#     Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
#     Saturn=[propagation_setup.acceleration.point_mass_gravity()],
#     Earth=[propagation_setup.acceleration.point_mass_gravity()],
# )
# NOTE
acceleration_settings = {}
for body in batch.MPC_objects:
    acceleration_settings[str(body)] = allAccels

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, batch.MPC_objects, central_bodies
)

# SPICE
initial_state_real = spice.get_body_cartesian_state_at_epoch(bodySpiceName, central_bodies[0], "J2000", "NONE", epoch_end)
print(initial_state_real)
# JPL HORIZONS
# hor = [-3.605416906345174E+08, -1.531951898772203E+08, 9.153673811646940E+05, 5.625558318778395E+00, -1.591561465522232E+01, -8.650255447335656E+00]
# initial_state_real = np.array(hor)*1000

initial_state = np.array(initial_state_real)
initial_state[0:3] += + np.random.rand(3)*1000
initial_state[3:] += + np.random.rand(3)*10


termination_condition = propagation_setup.propagator.time_termination(epoch_start)


dt = -60000
# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    epoch_end, dt, propagation_setup.integrator.rkf_78, dt, dt, 1.0, 1.0
)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies=central_bodies,
    acceleration_models=acceleration_models,
    bodies_to_integrate=batch.MPC_objects,
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

# #################################################################

estimator = numerical_simulation.Estimator(
    bodies=bodies,
    estimated_parameters=parameters_to_estimate,
    observation_settings=observation_settings_list,
    propagator_settings=propagator_settings,
    integrate_on_creation=True, 
)


pod_input = estimation.EstimationInput(
    observations_and_times=observation_collection,
    convergence_checker=estimation.estimation_convergence_checker(
        maximum_iterations=3,
    )
)

pod_input.define_estimation_settings(reintegrate_variational_equations=True)
print("running estimation")
pod_output = estimator.perform_estimation(pod_input)
results = parameters_to_estimate.parameter_vector
results = pod_output.parameter_history[:, -1]
print(results)
# ##################################################

# RESULTS
print("final output:")
print(pod_output)
print(results)

print("\n")
print(f"EPOCH: {epoch_end}")
print("\n")
print("ERROR TO SPICE:")
print("metres:")
spic_init = spice.get_body_cartesian_state_at_epoch(bodySpiceName, central_bodies[0], "J2000", "NONE", epoch_end)
print(np.array(results)-spic_init)
print("percentages:")
print(((np.array(results)-spic_init)/spic_init)*100)
print(f"Radial error: {np.sqrt(np.square((np.array(results) - spic_init)[0:3]).sum())/1000} km")
# print("\n")
# print("ERROR TO HORIZONS")
# hor = [-3.605416906345174E+08, -1.531951898772203E+08, 9.153673811646940E+05, 5.625558318778395E+00, -1.591561465522232E+01, -8.650255447335656E+00]
# hor = np.array(hor)*1000
# print("metres:")
# print(np.array(results) - hor)
# print("percentages:")
# print(((np.array(results)-hor)/hor)*100)
# print(f"Radial error: {np.sqrt(np.square((np.array(results) - hor)[0:3]).sum())/1000} km")
# print(
#     f"a = {(keplerElements[0] - (2.6962986 * 149.6e9))/149.6e9} AU | {(keplerElements[0] - (2.6962986 * 149.6e9))/1000} km"
# )
# print(f"e = {keplerElements[1]-0.1210537}")
# print(f"i = {np.degrees(keplerElements[2]) - 6.41021} deg")
# print(f"omega = {np.degrees(keplerElements[3])-124.96775} deg")
# # print(f"RAAN = {np.degrees(keplerElements[4])} deg")
# # print(f"theta = {np.degrees(keplerElements[5])} deg")

residual_history = pod_output.residual_history
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 11))
subplots_list = [ax1, ax2, ax3, ax4, ax5, ax6]


for i in range(residual_history.shape[1]):
    subplots_list[i].scatter(observation_collection.concatenated_times, residual_history[:, i])
plt.tight_layout()
fig.savefig("residuals")
# COMPARISON
# https://www.minorplanetcenter.net/db_search/show_object?utf8=%E2%9C%93&object_id=123


# ######################
propagator_settings = propagation_setup.propagator.translational(
    central_bodies=central_bodies,
    acceleration_models=acceleration_models,
    bodies_to_integrate=batch.MPC_objects,
    initial_states=pod_output.parameter_history[:, -1],
    initial_time=epoch_end,
    integrator_settings=integrator_settings,
    termination_settings=termination_condition,
)
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)
propagation_results = dynamics_simulator.propagation_results
state_history = propagation_results.state_history

# print(state_history)

times = list(state_history.keys())
states = np.array(list(state_history.values()))
spiceState = []
for t in times:
    s = spice.get_body_cartesian_state_at_epoch(bodySpiceName, central_bodies[0], "J2000", "NONE", t)
    spiceState.append(s)

spiceState = np.array(spiceState)

fig, ax = plt.subplots(3, 1, figsize=(9,10), sharex=True)
print(states.shape)
ax[0].plot(times, states[:, 0], label="propagated estimation")
ax[0].plot(times, spiceState[:, 0], label="direct spice")
ax[1].plot(times, states[:, 1])
ax[1].plot(times, spiceState[:, 1])
ax[2].plot(times, states[:, 2])
ax[2].plot(times, spiceState[:, 2])

ax[0].set_ylabel("X [m]")
ax[1].set_ylabel("Y [m]")
ax[2].set_ylabel("Z [m]")
ax[2].set_xlabel("time [s]")

ax[0].legend()
ax[0].grid()
ax[1].grid()
ax[2].grid()
fig.savefig("positons")


fig, ax = plt.subplots(3, 1, figsize=(9,10), sharex=True)
ax[0].plot(times, states[:, 0]-spiceState[:, 0], label="propagated - spice error")
ax[1].plot(times, states[:, 1]-spiceState[:, 1], label="propagated - spice error")
ax[2].plot(times, states[:, 2]-spiceState[:, 2], label="propagated - spice error")

ax[0].set_ylabel("X [m]")
ax[1].set_ylabel("Y [m]")
ax[2].set_ylabel("Z [m]")
ax[2].set_xlabel("time [s]")

ax[0].legend()
ax[0].grid()
ax[1].grid()
ax[2].grid()
fig.savefig("positonserror")