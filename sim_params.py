import numpy as np

from run_various_sims import *


######################################################################
#                   SIMULATION PARAMETERS                            #
######################################################################


''' Physical constants '''
G_M = 6.67430e-11 * 5.972e24
CONSTANT_G = 9.81
RADIUS_EARTH = 6.371e6


''' Simulation Metrics Names '''
INIT_ANGLE = 'init_angle'
INIT_VELOCITY = 'init_velocity' 
TIMES = 'times'
PATH_X = 'path_x'
PATH_Y = 'path_y'
VELOCITIES = 'velocities'
ACCELERATIONS = 'accelerations'


''' System variables (indices in the System vector)  
    To characterize the system we need to define the following variables:'''
X, Y, VX, VY = 0, 1, 2, 3


''' Other System Metrics. (indices in the Metrics vector)'''
V, A, ACC_HORIZ_DIST = 0, 1, 2



class Params:
    ''' inside a class to be easier to pass as a parameter to the functions'''
    def __init__(self):

        # Simulation details
        self._show_details = True
        self._sims_to_show_in_plot_metrics = 10
        self._dt = 0.01
        self._sim_max_time = 60 * 30

        # Initial conditions
        self._x_0 = 0
        self._altitude_0 = 130_000
        self._init_angles = np.negative(np.arange(start=0, stop=15.1, step=0.5))  # Angles in degrees --> we negate them because the path angle is measured down from the horizon
        self._init_velocities = np.arange(start=0, stop=15_001, step=300)    # Possible Initial velocities (m/s)

        # Capsule parameters
        self._capsule_mass = 12_000
        self._capsule_surface_area = 4 * np.pi
        self._capsule_drag_coefficient = 1.2
        self._capsule_lift_coefficient = 1

        # Parachute parameters
        self._parachute_surface_area = 301
        self._parachute_drag_coefficient = 1
        self._parachute_max_open_altitude = 1_000
        self._parachute_max_open_velocity = 100

        # Parameter boundaries
        self._min_horizontal_distance = 2_500_000
        self._max_horizontal_distance = 4_500_000
        self._max_landing_velocity = 25
        self._max_acceleration = 150
    

    def __str__(self):
        return f"Params: \n" \
               f"dt = {self._dt}\n" \
               f"sim_max_time = {self._sim_max_time}\n" \
               f"x_0 = {self._x_0}\n" \
               f"altitude_0 = {self._altitude_0}\n" \
               f"init_angles = {self._init_angles}\n" \
               f"init_velocities = {self._init_velocities}\n" \
               f"capsule_mass = {self._capsule_mass}\n" \
               f"capsule_surface_area = {self._capsule_surface_area}\n" \
               f"capsule_drag_coefficient = {self._capsule_drag_coefficient}\n" \
               f"capsule_lift_coefficient = {self._capsule_lift_coefficient}\n" \
               f"parachute_surface_area = {self._parachute_surface_area}\n" \
               f"parachute_drag_coefficient = {self._parachute_drag_coefficient}\n" \
               f"parachute_max_open_altitude = {self._parachute_max_open_altitude}\n" \
               f"parachute_max_open_velocity = {self._parachute_max_open_velocity}\n" \
               f"min_horizontal_distance = {self._min_horizontal_distance}\n" \
               f"max_horizontal_distance = {self._max_horizontal_distance}\n" \
               f"max_landing_velocity = {self._max_landing_velocity}\n" \
               f"max_acceleration = {self._max_acceleration}\n" \
               f"show_details = {self._show_details}\n" \
               f"sims_to_show_in_plot_metrics = {self._sims_to_show_in_plot_metrics}\n"



def get_params():
    p = Params()
    if SIM_TO_RUN == REENTRY_SIM:
        if SIM_TYPE == NORMAL_SIM:
            return Params()
    if SIM_TO_RUN == PROJECTILE_SIM:
        if SIM_TYPE == NORMAL_SIM:
            p._altitude_0 = 0
            p._init_angles = [30,45,60]
            p._init_velocities = [100]
            p._capsule_mass = 1000
            p._capsule_surface_area = 1
            p._capsule_drag_coefficient = 0
            p._capsule_lift_coefficient = 0
            return p
        

