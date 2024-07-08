import numpy as np


######################################################################
#                   CHOOSE SIMULATION OPTIONS                        #
# sim_params are set according with the choosen options bellow       #
######################################################################


''' 1. Choose the simulation to run from options below '''
SIM_TO_RUN = 1
#------------------------
REENTRY_SIM = 1
PROJECTILE_SIM = 2          # simulation of a projectile being launched with different angles and velocities
#------------------------        


''' 2. Choose type of simulation from options below: '''
SIM_TYPE = 1
#------------------------
NORMAL_SIM = 1                  # we'll start simulation for several angles and velocities
NORMAL_SIM_BUT_LESS_PAIRS = 2   # we'll start simulation for less angles and velocities
VERTICAL_SIM = 3                # we'll start the simulation without velocity, so with forces object will move vertically
                                # TODO: If Lift = TRUE ->  For vertical simulation, make sure LIFT = 0, because if not there will be horizontal movement; try with lift = 0 and = 1 to see the lift effect
ORBITAL_VEL_SIM = 4             # if with ROUND_EARTH = TRUE, we'll start the simulation with the orbital velocity, so the object will keep the same altitude and will move horizontally
ESCAPE_VEL_SIM = 5              # if with ROUND_EARTH = TRUE, we'll start the simulation with the escape velocity, so the object will keep the same altitude and will move horizontally
#------------------------



''' 3. Choose more options: '''
SIM_WITH_PARACHUTE = True          # if True we'll simulate the reentry with deployment of the parachutes after some conditions are met

ROUND_EARTH = False                  # if True we'll simulate the reentry in a round Earth

LIFT_PERPENDICULAR_TO_VELOCITY = True  # if False, all lift force will be added to y component regardless of velocity direction
                                        # if True, lift force will be perpendicular to velocity direction, and always pointing up


''' If you want different values than the default ones, choose here: '''
DT = 0.01                        # time steps (s)
SIM_MAX_TIME = 60 * 30            # max time for the simulation (s)
SIMS_TO_SHOW_IN_PLOT_METRICS = 10 # number of simulations to show in the plot metrics (we don't show all of them to not clutter the plot)

CAPSULE_DRAG_COEFFICIENT = 1.2      # drag coefficient to use in the simulation
CAPSULE_LIFT_COEFFICIENT = 1        # lift coefficient to use in the simulation

PARACHUTE_DRAG_COEFFICIENT = 1      # drag coefficient to use in the simulation
PARACHUTE_MAX_OPEN_ALTITUDE = 1_000 # maximum altitude to open the parachute

NEWTON_EPSILON = 0.0001   # for newton method (implicit) - to stop iterating when we find a "almost root" smaller than this value
NEWTON_MAX_ITER = 1000    # for newton method (implicit) - maximum number of iterations

USE_CUSTOMIZED_INIT_VALUES = True
INIT_ANGLES = [ -8, -16]    # initial angles (degrees)
INIT_VELOCITIES = [4_000, 8_000, 12_000, 16_000] # initial velocities (m/s)



######################################################################
#                   SIMULATION PARAMETERS                            #
######################################################################


''' Physical constants '''
G_M = 6.67430e-11 * 5.972e24
CONSTANT_G = 9.81
RADIUS_EARTH = 6.371e6
CONSTANT_AIR_DENSITY = 1.225


''' Simulation Metrics Names '''
INIT_ANGLE = 'init_angle'
INIT_VELOCITY = 'init_velocity' 
TIMES = 'times'
PATH_X = 'path_x'
PATH_Y = 'path_y'
ABS_VELOCITIES = 'velocities'
Y_VELOCITIES = 'y_velocities'
ACCELERATIONS = 'accelerations'
CHUTE_OPENING = 'chute_opening'


''' System variables (indices in the System vector)  
    To characterize the system we need to define the following variables:'''
X, Y, VX, VY = 0, 1, 2, 3


''' Other System Metrics. (indices in the Metrics vector)'''
V, A, EARTH_ANGLE, CHUTE_OPEN = 0, 1, 2, 3



class Params:
    ''' inside a class to be easier to pass as a parameter to the functions'''
    def __init__(self):

        # Simulation details
        self.dt = DT
        self.sim_max_time = SIM_MAX_TIME
        self.is_reentry_sim = SIM_TO_RUN == REENTRY_SIM
        self.sim_with_parachute = SIM_WITH_PARACHUTE        
        self.sims_to_show_in_plot_metrics = SIMS_TO_SHOW_IN_PLOT_METRICS
        self.epsilon = NEWTON_EPSILON
        self.max_iter = NEWTON_MAX_ITER
        self.sim_round_earth = ROUND_EARTH
        self.lift_perpendicular_to_velocity = LIFT_PERPENDICULAR_TO_VELOCITY
        self.orbit_or_escape_vel_sim = False

        # Initial conditions
        self.x_0 = 0
        self.altitude_0 = 130_000
        self.init_angles = INIT_ANGLES if USE_CUSTOMIZED_INIT_VALUES else np.negative(np.arange(start=0, stop=15.1, step=0.5))  # Angles in degrees --> we negate them because the path angle is measured down from the horizon
        self.init_velocities = INIT_VELOCITIES if USE_CUSTOMIZED_INIT_VALUES else np.arange(start=0, stop=15_001, step=300)    # Possible Initial velocities (m/s)

        # Capsule parameters
        self.capsule_mass = 12_000
        self.capsule_surface_area = 4 * np.pi
        self.capsule_drag_coefficient = CAPSULE_DRAG_COEFFICIENT
        self.capsule_lift_coefficient = CAPSULE_LIFT_COEFFICIENT

        # Parachute parameters
        self.parachute_surface_area = 301
        self.parachute_drag_coefficient = PARACHUTE_DRAG_COEFFICIENT
        self.parachute_max_open_altitude = PARACHUTE_MAX_OPEN_ALTITUDE
        self.parachute_max_open_velocity = 100

        # Parameter boundaries
        self.min_horizontal_distance = 2_500_000
        self.max_horizontal_distance = 4_500_000
        self.max_landing_velocity = 25
        self.max_acceleration = 150
    

    def __str__(self):
        return f"Params: \n" \
               f"dt = {self.dt}\n" \
               f"sim_max_time = {self.sim_max_time}\n" \
               f"x_0 = {self.x_0}\n" \
               f"altitude_0 = {self.altitude_0}\n" \
               f"init_angles = {self.init_angles}\n" \
               f"init_velocities = {self.init_velocities}\n" \
               f"capsule_mass = {self.capsule_mass}\n" \
               f"capsule_surface_area = {self.capsule_surface_area}\n" \
               f"capsule_drag_coefficient = {self.capsule_drag_coefficient}\n" \
               f"capsule_lift_coefficient = {self.capsule_lift_coefficient}\n" \
               f"parachute_surface_area = {self.parachute_surface_area}\n" \
               f"parachute_drag_coefficient = {self.parachute_drag_coefficient}\n" \
               f"parachute_max_open_altitude = {self.parachute_max_open_altitude}\n" \
               f"parachute_max_open_velocity = {self.parachute_max_open_velocity}\n" \
               f"min_horizontal_distance = {self.min_horizontal_distance}\n" \
               f"max_horizontal_distance = {self.max_horizontal_distance}\n" \
               f"max_landing_velocity = {self.max_landing_velocity}\n" \
               f"max_acceleration = {self.max_acceleration}\n" \
               f"sims_to_show_in_plot_metrics = {self.sims_to_show_in_plot_metrics}\n" \
               f"epsilon = {self.epsilon}\n" \
               f"max_iter = {self.max_iter}\n" \
               f"sim_round_earth = {self.sim_round_earth}\n" \
               f"lift_perpendicular_to_velocity = {self.lift_perpendicular_to_velocity}\n"


def correct_exception_params(p: Params):
    if p.capsule_mass == 0:
        p.capsule_mass = 1e-20
    return p

def orbital_velocity(altitude):
    return np.sqrt(G_M / (RADIUS_EARTH + altitude))

def get_params():
    p = Params()
    if SIM_TO_RUN == REENTRY_SIM:
        if SIM_TYPE == NORMAL_SIM:
            return correct_exception_params(p)
        elif SIM_TYPE == NORMAL_SIM_BUT_LESS_PAIRS:
            p.init_angles = [-8, -16, -32]
            p.init_velocities = [0, 3_000, 6_000] if ROUND_EARTH else [0, 6_000, 15_000] # with round earth we reduce speed to not escape to space
            return correct_exception_params(p)
        elif SIM_TYPE == VERTICAL_SIM:
            p.x_0 = 100_000
            p.init_angles = [90]
            p.init_velocities = [0]
            return correct_exception_params(p)
        elif SIM_TYPE == ORBITAL_VEL_SIM:
            p.sim_max_time = 60 * 4
            p.init_angles = [0]
            p.init_velocities = [orbital_velocity(p.altitude_0)]
            p.orbit_or_escape_vel_sim = True
            return correct_exception_params(p)
        else: # SIM_TYPE == ESCAPE_VEL_SIM:
            p.sim_max_time = 60 * 10
            p.init_angles = [-0.3]
            p.init_velocities = [ 10 * orbital_velocity(p.altitude_0)]
            p.orbit_or_escape_vel_sim = True
            return correct_exception_params(p)
        

    if SIM_TO_RUN == PROJECTILE_SIM:
        if SIM_TYPE == NORMAL_SIM:
            p.altitude_0 = 0
            p.init_angles = [30, 45, 60]
            p.init_velocities = [1000]
            p.capsule_mass = 1
            p.capsule_surface_area = 1
            p.capsule_drag_coefficient = 0.01
            p.capsule_lift_coefficient = 0
            p.parachute_drag_coefficient = 0
            return correct_exception_params(p)
        

