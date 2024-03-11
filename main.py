# Load standard modules
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro.element_conversion import keplerian_to_cartesian

import pycode.argos as argos
import pycode.plot_gen as plot_gen

import os

#######################
# Input Section Start #
#######################

## Configuration

orbit_name = 'new_orbit'

time_between_measurements = 1  # seconds don't go under 10 sec for one month of propagation otherwise it will take forever!!
orbit_apoapsis = 550  # km
orbit_periapsis = 200  # km
orbit_inclination = 81  # deg

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2029, 9, 1).epoch()
simulation_end_epoch   = DateTime(2029, 10, 1).epoch()

#######################
# Input Section Stop  #
#######################

# Create output directory
os.makedirs(os.path.join("data", orbit_name), exist_ok=True)
os.makedirs(os.path.join("plots", orbit_name), exist_ok=True)

## Environment setup

### Create the bodies

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus"]

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Create default body settings, usually from `spice`.
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Use nrlmsise00
body_settings.get( "Earth" ).atmosphere_settings = environment_setup.atmosphere.nrlmsise00()

# Create system of selected celestial bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

### Create the vehicle
"""
Let's now create the 400kg satellite for which the perturbed orbit around Earth will be propagated.
"""

# Create vehicle objects.
bodies.create_empty_body("FranzSat")

bodies.get("FranzSat").mass = 60.0

# Create aerodynamic coefficient interface settings, and add to vehicle
reference_area = 0.4  # Average projection area of the sat
# reference_area = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coefficient = 2.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area, [drag_coefficient, 0, 0]
)
environment_setup.add_aerodynamic_coefficient_interface(
    bodies, "FranzSat", aero_coefficient_settings)

# To account for the pressure of the solar radiation on the satellite, let's add another interface. This takes a radiation pressure coefficient of 1.2, and a radiation area of 4m$^2$. This interface also accounts for the variation in pressure cause by the shadow of Earth.

# Create radiation pressure settings, and add to vehicle
reference_area_radiation = 0.4  # Average projection area of the sat
# reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
radiation_pressure_coefficient = 2.2
occulting_bodies_dict = dict()
occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )
environment_setup.add_radiation_pressure_target_model(
    bodies, "FranzSat", vehicle_target_settings)

## Propagation setup

# Define bodies that are propagated
bodies_to_propagate = ["FranzSat"]

# Define central bodies of propagation
central_bodies = ["Earth"]

### Create the acceleration model

# Define accelerations acting on Delfi-C3 by Sun and Earth.
accelerations_settings_delfi_c3 = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
        propagation_setup.acceleration.aerodynamic()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    # Mars=[
    #     propagation_setup.acceleration.point_mass_gravity()
    # ],
    # Venus=[
    #     propagation_setup.acceleration.point_mass_gravity()
    # ]
)

# Create global accelerations settings dictionary.
acceleration_settings = {"FranzSat": accelerations_settings_delfi_c3}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

### Define the initial state

# Retrieve the initial state of Delfi-C3 using Two-Line-Elements (TLEs)

# Orbital elements
Re = 6371000  # Earth radius in m
aph = orbit_apoapsis * 10**3 + Re  # m
per = orbit_periapsis * 10**3 + Re  # m
sma = (aph+per)/2  # semi-major axis - m
e = (aph-per)/(aph+per)  # eccentricity
i = np.deg2rad(orbit_inclination)  # inclination - rad
ta = np.deg2rad(0)  # true anomaly - rad
raan = np.deg2rad(0)  # right ascension of ascending node - rad
aop = np.deg2rad(0)  # argument of periapsis - rad
# Definition of initial state
mu_e = body_settings.get('Earth').gravity_field_settings.gravitational_parameter
initial_kepl = np.array([sma, e, i, aop, raan, ta])
initial_state = keplerian_to_cartesian(initial_kepl, mu_e)

### Define dependent variables to save


# Define list of dependent variables to save
dependent_variables_to_save = [
    # propagation_setup.dependent_variable.total_acceleration("Delfi-C3"),
    propagation_setup.dependent_variable.keplerian_state("FranzSat", "Earth"),
    propagation_setup.dependent_variable.latitude("FranzSat", "Earth"),
    propagation_setup.dependent_variable.longitude("FranzSat", "Earth"),
    propagation_setup.dependent_variable.altitude("FranzSat", "Earth")
]

### Create the propagator settings

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
# fixed_step_size = 10.0
fixed_step_size = time_between_measurements
integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save
)

## Propagate the orbit
"""
The orbit is now ready to be propagated.

This is done by calling the `create_dynamics_simulator()` function of the `numerical_simulation module`.
This function requires the `bodies` and `propagator_settings` that have all been defined earlier.

After this, the history of the propagated state over time, containing both the position and velocity history, is extracted.
This history, taking the form of a dictionary, is then converted to an array containing 7 columns:
- Column 0: Time history, in seconds since J2000.
- Columns 1 to 3: Position history, in meters, in the frame that was specified in the `body_settings`.
- Columns 4 to 6: Velocity history, in meters per second, in the frame that was specified in the `body_settings`.

The same is done with the dependent variable history. The column indexes corresponding to a given dependent variable in the `dep_vars` variable are printed when the simulation is run, when `create_dynamics_simulator()` is called.
Do mind that converting to an ndarray using the `result2array()` utility will shift these indexes, since the first column (index 0) will then be the times.
"""

# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state and depedent variable history and convert it to an ndarray
states = dynamics_simulator.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.dependent_variable_history
dep_vars_array = result2array(dep_vars)

## Post-process the propagation results

# ### Total acceleration over time
# """
# Let's first plot the total acceleration on the satellite over time. This can be done by taking the norm of the first three columns of the dependent variable list.
# """

# # Plot total acceleration as function of time
# time_hours = dep_vars_array[:,0]/3600
# total_acceleration_norm = np.linalg.norm(dep_vars_array[:,1:4], axis=1)
# plt.figure(figsize=(9, 5))
# plt.title("Total acceleration norm on Delfi-C3 over the course of propagation.")
# plt.plot(time_hours, total_acceleration_norm)
# plt.xlabel('Time [hr]')
# plt.ylabel('Total Acceleration [m/s$^2$]')
# plt.xlim([min(time_hours), max(time_hours)])
# plt.grid()
# plt.tight_layout()

### Ground track
"""
Let's then plot the ground track of the satellite in its first 3 hours. This makes use of the latitude and longitude dependent variables.
"""

# Plot ground track for a period of 3 hours
time_hours = (dep_vars_array[:,0]-dep_vars_array[0,0])/3600  # Time in hours since beginning of propagation
kepler_elements = dep_vars_array[:,1:7]  # Keplerian elements in m and rad
latitude = dep_vars_array[:,7]  # Latitude in rad
longitude = dep_vars_array[:,8]  # Longitude in rad
altitude = dep_vars_array[:,9]  # Altitude in m

# Save arrays

np.savez_compressed(os.path.join("data", orbit_name, orbit_name), alt=altitude, lon=longitude, lat=latitude, time=time_hours)  # 0: time, 1-3: pos in m, 4-6: vel in m/s

# argos.nparray_saver_npz(os.path.join(orbit_name, 'altitude'),altitude)
# argos.nparray_saver_npz(os.path.join(orbit_name, 'longitude'),longitude)
# argos.nparray_saver_npz(os.path.join(orbit_name, 'latitude'),latitude)
# argos.nparray_saver_npz(os.path.join(orbit_name, 'kepler_elements'),kepler_elements)
# argos.nparray_saver_npz(os.path.join(orbit_name, 'time_hours'),time_hours)
# argos.nparray_saver_npz(os.path.join(orbit_name, 'cartesian_states'),states_array)  # 0: time, 1-3: pos in m, 4-6: vel in m/s

# argos.nparray_saver(os.path.join(orbit_name, 'altitude'),altitude)
# argos.nparray_saver(os.path.join(orbit_name, 'longitude'),longitude)
# argos.nparray_saver(os.path.join(orbit_name, 'latitude'),latitude)
# argos.nparray_saver(os.path.join(orbit_name, 'kepler_elements'),kepler_elements)
# argos.nparray_saver(os.path.join(orbit_name, 'time_hours'),time_hours)
# argos.nparray_saver(os.path.join(orbit_name, 'cartesian_states'),states_array)  # 0: time, 1-3: pos in m, 4-6: vel in m/s

# argos.nparray_saver('altitude',altitude)
# argos.nparray_saver('latitude',latitude)
# argos.nparray_saver('kepler_elements',kepler_elements)
# argos.nparray_saver('tiome_hours',time_hours)
# argos.nparray_saver('cartesian_states',states_array)  # 0: time, 1-3: pos in m, 4-6: vel in m/s

# Ground track plot
fig_track = plot_gen.track(latitude, longitude)
argos.fig_saver(os.path.join(orbit_name, 'ground_track'),fig_track)
# Plot Altitude in Time
fig_alt = plt.figure(figsize=(9, 5))
plt.title("Altitude of FranzSat")
plt.plot(time_hours, altitude/1000)
plt.xlabel('Time [hours]')
plt.ylabel('Altitude [km]')
plt.grid()
plt.tight_layout()
argos.fig_saver(os.path.join(orbit_name, 'altitude'), fig_alt)



# hours = 24*3
# subset = int(len(time_hours) / 24 * hours)
# latitude = np.rad2deg(latitude[0: subset])
# longitude = np.rad2deg(longitude[0: subset])
# fig = plt.figure(figsize=(9, 5))
# plt.title("Ground track of FranzSat")
# plt.scatter(longitude, latitude, s=1)
# plt.xlabel('Longitude [deg]')
# plt.ylabel('Latitude [deg]')
# plt.xlim([-180, 180])
# plt.ylim([-90, 90])
# plt.xticks(np.arange(-180, 181, step=45))
# plt.yticks(np.arange(-90, 91, step=45))
# plt.grid()
# plt.tight_layout()
# plt.show()

# ## Plot the orbit
# pos = states_array[:,1:4]
# fig_orbit = plot_gen.orbit(pos)
# # argos.fig_saver('orbit', fig_orbit)

# ## Plot Altitude in time
# plt.figure(figsize=(9, 5))
# plt.title("Altitude of FranzSat")
# plt.plot(time_hours, altitude)
# plt.xlabel('Time [hours]')
# plt.ylabel('Altitude [km]')
# plt.grid()
# plt.tight_layout()
# # plt.show()


### Kepler elements over time
"""
Let's now plot each of the 6 Kepler element as a function of time, also as saved in the dependent variables.
"""

# Plot Kepler elements as a function of time
fig_kepl, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig_kepl.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Semi-major Axis
semi_major_axis = kepler_elements[:,0] / 1e3
ax1.plot(time_hours, semi_major_axis)
ax1.set_ylabel('Semi-major axis [km]')

# Eccentricity
eccentricity = kepler_elements[:,1]
ax2.plot(time_hours, eccentricity)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:,2])
ax3.plot(time_hours, inclination)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(np.unwrap(kepler_elements[:,3]))
ax4.plot(time_hours, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_elements[:,4])
ax5.plot(time_hours, raan)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_elements[:,5])
ax6.scatter(time_hours, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig_kepl.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
plt.tight_layout()
argos.fig_saver(os.path.join(orbit_name, 'kepler_elements'), fig_kepl)


# ### Accelerations over time
# """
# Finally, let's plot and compare each of the included accelerations.
# """

# plt.figure(figsize=(9, 5))

# # Point Mass Gravity Acceleration Sun
# acceleration_norm_pm_sun = dep_vars_array[:,12]
# plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Sun')

# # Point Mass Gravity Acceleration Moon
# acceleration_norm_pm_moon = dep_vars_array[:,13]
# plt.plot(time_hours, acceleration_norm_pm_moon, label='PM Moon')

# # Point Mass Gravity Acceleration Mars
# acceleration_norm_pm_mars = dep_vars_array[:,14]
# plt.plot(time_hours, acceleration_norm_pm_mars, label='PM Mars')

# # Point Mass Gravity Acceleration Venus
# acceleration_norm_pm_venus = dep_vars_array[:,15]
# plt.plot(time_hours, acceleration_norm_pm_venus, label='PM Venus')

# # Spherical Harmonic Gravity Acceleration Earth
# acceleration_norm_sh_earth = dep_vars_array[:,16]
# plt.plot(time_hours, acceleration_norm_sh_earth, label='SH Earth')

# # Aerodynamic Acceleration Earth
# acceleration_norm_aero_earth = dep_vars_array[:,17]
# plt.plot(time_hours, acceleration_norm_aero_earth, label='Aerodynamic Earth')

# # Cannonball Radiation Pressure Acceleration Sun
# acceleration_norm_rp_sun = dep_vars_array[:,18]
# plt.plot(time_hours, acceleration_norm_rp_sun, label='Radiation Pressure Sun')

# plt.xlim([min(time_hours), max(time_hours)])
# plt.xlabel('Time [hr]')
# plt.ylabel('Acceleration Norm [m/s$^2$]')

# plt.legend(bbox_to_anchor=(1.005, 1))
# plt.suptitle("Accelerations norms on Delfi-C3, distinguished by type and origin, over the course of propagation.")
# plt.yscale('log')
# plt.grid()
# plt.tight_layout()
