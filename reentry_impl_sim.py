import numpy as np
import pandas as pd
import plot_functions as plot
from scipy.interpolate import CubicSpline


LIFT_PERPENDICULAR_TO_VELOCITY = False  # if False, all lift force will be added to y component regardless of velocity direction
                                        # if True, lift force will be perpendicular to velocity direction, and always pointing up
CONSTANT_GRAVITY = False                # if True we'll use constant values for gravity
CONSTANT_AIR_DENSITY = False            # if True we'll use constant values for air density
SIM_WITH_PARACHUTE = True               # if True we'll simulate the reentry with deployment of the parachutes after some conditions are met

SHOW_DETAILS = False

dt = 0.5                                # time steps (s)
SIM_MAX_TIME = 5_000                    # max time for the simulation (s)

SIM_TO_SHOW_IN_PLOT_METRICS = 10        # number of simulations to show in the plot metrics (we don't show all of them to not clutter the plot)

############################################################################################################
#                                   CONSTANTS AND PARAMETERS
############################################################################################################

# Physical constants
G_M = 6.67430e-11 * 5.972e24    # G (Gravitational constant) * Earth mass, m^3 kg^-1 s^-2
CONSTANT_G = 9.81               # constant gravity (m/s^2)
RADIUS_EARTH = 6.371e6          # Earth radius (m)

''' Reentry Simulation Parameters - to try different initial angles and velocities and find the best combinations '''

X_0 = 0                                                          # Initial x position (m)
ALTITUDE_0 = 130_000                                                # "interface" == Initial altitude (m)
# TODO: meter todas as velocidades e angulos
INIT_VELOCITIES = np.arange(start=0, stop=15_001, step=2_000)    # Possible Initial velocities (m/s)
INIT_ANGLES = np.negative (np.arange(start=0, stop=15.1, step=2))  # Angles in degrees --> we negate them because the path angle is measured down from the horizon

# Capsule parameters
CAPSULE_MASS = 12_000               # Mass of the capsule (kg)
CAPSULE_SURFACE_AREA = 4 * np.pi    # surface considered for drag coefficient (m^2)
CAPSULE_DRAG_COEFFICIENT = 1.2
CAPSULE_LIFT_COEFFICIENT = 1

# Parachute parameters
PARACHUTE_SURFACE_AREA = 301                # surface considered for parachute's drag coefficient (m^2)
PARACHUTE_DRAG_COEFFICIENT = 1.0            # parachute's drag coefficient
PARACHUTE_MAX_OPEN_ALTITUDE = 1_000                   # boundary altitude for deployment of the parachutes (m)
PARACHUTE_MAX_OPEN_VELOCITY = 1_00                    # boundary velocity for deployment of the parachutes (m/s)

# Parameter boundaries
MIN_HORIZONTAL_DISTANCE = 2_500_000         # lower boundary for horizontal distance (m)
MAX_HORIZONTAL_DISTANCE = 4_500_000         # higher boundary for horizontal distance (m)
MAX_LANDING_VELOCITY = 25                   # final boundary velocity for deployment (m/s)
MAX_ACCELERATION = 150                      # acceleration boundary for the vessel and crew (m/s^2)


############################################################################################################
#                                   AIR DENSITY 
############################################################################################################
DENSITY_CSV = pd.read_csv('air_density.csv')                # Air density table
ALTITUDE = DENSITY_CSV['altitude']                          # Altitude values
AIR_DENSITY = DENSITY_CSV['air_density']                    # Air density values
f = CubicSpline(ALTITUDE, AIR_DENSITY, bc_type='natural')   # Cubic spline interpolation for air density


############################################################################################################
#                                   IMPLICIT SIMULATION FUNCTIONS
############################################################################################################
# x_f = 
# y_f = 
# vx_f = 
# vy_f = 



def make_round_earth (x, y, x_step, y_step, earth_angle):
    ''' given x and y variables (e.g. position or velocities, or accelaration...),
        and the step in each direction in the flat earth, and the earth angle (angle in origin from y axis to current position), 
        converts flat steps to round earth steps and adds it to the x, y variables. '''
    x += y_step * np.sin(earth_angle) + x_step * np.cos(earth_angle)  
    y += y_step * np.cos(earth_angle) - x_step * np.sin(earth_angle)  # y_flat is decreasing because we are going down
    return x, y


def get_air_density_cubic_spline(y):
    altitude = y - RADIUS_EARTH
    result = f(altitude)
    return result if result > 0.0 else 0.0


def get_tot_acceleration(x, y, vx, vy):
    '''calculates the total acceleration on the capsule, which depends if the parachutes are deployed or not.'''
    # Common air components for drag and lift
    v_abs = np.sqrt(vx**2 + vy**2)
    air_density =  1.225 if CONSTANT_AIR_DENSITY else get_air_density_cubic_spline(y)
    F_air = 0.5 * CAPSULE_SURFACE_AREA * air_density * v_abs / CAPSULE_MASS

    # Drag and Lift: 
    ax = -F_air * CAPSULE_DRAG_COEFFICIENT * vx # "-" because it's a resistance force on opposite direction of the velocity (in case of object falling down velocity is also negative, so in that case the force will be positive, slowing the falling)
    ay = -F_air * CAPSULE_DRAG_COEFFICIENT * vy 

    if SIM_WITH_PARACHUTE and v_abs <= PARACHUTE_MAX_OPEN_VELOCITY and (y - RADIUS_EARTH) <= PARACHUTE_MAX_OPEN_ALTITUDE:
        # if we open the parachutes, we'll add its drag force in the opposite direction of the velocity
        F_drag_parachute = 0.5 * PARACHUTE_DRAG_COEFFICIENT * PARACHUTE_SURFACE_AREA * air_density * v_abs / CAPSULE_MASS
        ax -= F_drag_parachute * vx
        ay -= F_drag_parachute * vy
    else:
        lift = np.abs(F_air * CAPSULE_LIFT_COEFFICIENT * v_abs)
       
        if LIFT_PERPENDICULAR_TO_VELOCITY:
            # Add lift components to y and x, in order to keep it perpendicular to velocity: for that we find the velocity angle, we find a lift angle (and we make sure it is always pointing up so if l_angle > 180º we subtract 180º), and we find components of lift for y and x:
            v_angle = np.arctan2(vy, vx)
            l_angle = v_angle + np.pi/2
            if l_angle > np.pi:
                l_angle -= np.pi
            ay += lift * np.sin(v_angle) 
            ax += lift * np.cos(v_angle) 
        else:
            # Add all lift force to y component, independent of the direction of velocity:
            ay += lift

    # Gravity
    g = CONSTANT_G if CONSTANT_GRAVITY else G_M / (x**2 + y**2) # G_M / r**2 # simplified from: (np.sqrt(x**2 + y**2)**2)
    ay -= g
    return ax, ay


def run_entry_implicit_simulation(angle_0, v_0, altitude_0 = ALTITUDE_0, x_0 = X_0):
    '''runs a simulation of the capsule reentry'''
    if SHOW_DETAILS:
        print("\nsimulation:  angle: ", angle_0, "   init velocity: ", v_0, "   altitude: ", altitude_0, "   x_0: ", x_0)

    # metrics to store
    times = []
    path_x = [] # we discard initial values for simplicity (because we don't know initial values of other metrics like acceleration)
    path_y = []
    velocities = []
    accelerations = []
    passed_max_g_limit = False

    # accumulator variables for the simulation
    time = 0
    x = x_0                         
    y = RADIUS_EARTH + altitude_0  
    earth_angle = np.arctan(x / y)
    sum_earth_angle = 0

    # initial velocity on flat earth
    angle_0_rad = np.radians(angle_0)
    vx = v_0 * np.cos(angle_0_rad)
    vy = v_0 * np.sin(angle_0_rad)
    if SHOW_DETAILS:
        print( "Starting loops with: x: ", x, "   y: ", y, " (R = ", RADIUS_EARTH,")   vx: ", vx, "   vy: ", vy)

    while y >= RADIUS_EARTH:
        if round(time) % 2_000 == 0: 
            if SHOW_DETAILS:
                print("time: ", round(time, 0), "   x: ", round(x, 0), " y: ", round(y, 0), " vx: ", round(vx, 0), " vy: ", round(vy, 0))
            if time > SIM_MAX_TIME:
                print("Max time surpassed. Exiting simulation for angle: ", angle_0, "   init velocity: ", v_0)
                break
        
        # time
        time += dt

        # acceleration
        ax, ay = get_tot_acceleration(x, y, vx, vy)

        a = np.sqrt(ax**2 + ay**2)
        if(a > MAX_ACCELERATION):
            passed_max_g_limit = True
        



def main():

    successful_pairs = []
    acceleration_pairs = []
    velocity_pairs = []
    distance_pairs = []

    if SHOW_DETAILS:
        axs = plot.start_sims_metrics_plot(True, SIM_TO_SHOW_IN_PLOT_METRICS)
        random_sim_to_show = np.random.randint(0, len(INIT_ANGLES)*len(INIT_VELOCITIES), size=SIM_TO_SHOW_IN_PLOT_METRICS) # we'll show 10 random simulations and not all of them to not clutter the plot
    sim_to_show = 0
    for angle_0 in INIT_ANGLES:
        for v_0 in INIT_VELOCITIES:
            sim_metrics, successfull_landing, g_limit, velocity_limit, horizontal_landing_limit = run_entry_implicit_simulation(angle_0, v_0)
            if successfull_landing:
                successful_pairs.append((angle_0, v_0))
            if g_limit:
                acceleration_pairs.append((angle_0, v_0))
            if velocity_limit:
                velocity_pairs.append((angle_0, v_0))
            if horizontal_landing_limit:
                distance_pairs.append((angle_0, v_0))
            if SHOW_DETAILS:
                sim_to_show += 1
                if sim_to_show in random_sim_to_show:
                    plot.plot_sim_metrics(axs, sim_metrics, True)
    if SHOW_DETAILS:
        plot.end_sims_metrics_plot()
    plot.plot_reentry_conditions(acceleration_pairs, velocity_pairs, distance_pairs)
    plot.plot_reentry_parameters(successful_pairs)



if __name__ == "__main__":
    main()