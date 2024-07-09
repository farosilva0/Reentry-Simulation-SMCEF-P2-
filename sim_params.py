import numpy as np


######################################################################
#                   CHOOSE SIMULATION OPTIONS                        #
#   sim_params are set according with the choosen options bellow     #
######################################################################

''' 1. Choose if you want to save plot images in folder "plot_images". '''
SAVE_PLOT_IMAGES = True
SIM_NAME_FOR_IMAGE = "implicit" 


''' 1. Choose the simulation to run from options below '''
SIM_TO_RUN = 2
# --------------------------------------------------------------------
# REENTRY_SIMULATION OPTIONS: 
REENTRY_SIM_NORMAL = 1                  # we'll start simulation for several angles and velocities
REENTRY_SIM_CUSTOMIZED_PAIRS = 2        # we'll run the simulation with the angles and velocities choosen bellow
REENTRY_SIM_VERTICAL_MOV = 3            # we'll start the simulation without velocity, so with forces object will move vertically
REENTRY_SIM_HORIZONTAL_MOV = 4          # we'll start the simulation with horizontal angle and initial velocity, and forces will be 0
REENTRY_SIM_ORBITAL_MOV = 5             # if with ROUND_EARTH = TRUE, we'll start the simulation with the orbital velocity, so the object will keep the same altitude and will move horizontally
REENTRY_SIM_ESCAPE_VEL_MOV = 6          # if with ROUND_EARTH = TRUE, we'll start the simulation with the escape velocity, so the object will keep the same altitude and will move horizontally
# PROJECTILE_SIMULATION OPTIONS: 
PROJECTILE_SIM = 7                      # we'll start simulation for several angles and velocities
# --------------------------------------------------------------------

''' 2. Choose more options: '''

# Pairs to run in REENTRY_SIM_CUSTOMIZED_PAIRS
INIT_ANGLES = [-1, -2, -4, -8] #[ -8, -16]    # initial angles (degrees)
INIT_VELOCITIES = [2_000, 4_000, 8_000] #, 12_000, 16_000] # initial velocities (m/s)

SIM_WITH_PARACHUTE = True          # if True we'll simulate the reentry with deployment of the parachutes after some conditions are met

LIFT_PERPENDICULAR_TO_VELOCITY = False  # if False, all lift force will be added to y component regardless of velocity direction
                                        # if True, lift force will be perpendicular to velocity direction, and always pointing up
MAX_ANGLE_OF_ATTACK = 0                     # angle of attack in degrees (0 means no angle of attack)

ROUND_EARTH = False                  # if True we'll simulate the reentry in a round Earth


DT = 0.01                        # time steps (s)
SIM_MAX_TIME = 60 * 30            # max time for the simulation (s)
SIMS_TO_SHOW_IN_PLOT_METRICS = 12 # number of simulations to show in the plot metrics (we don't show all of them to not clutter the plot)

CAPSULE_DRAG_COEFFICIENT = 1.2      # drag coefficient to use in the simulation
CAPSULE_LIFT_COEFFICIENT = 1        # lift coefficient to use in the simulation

PARACHUTE_DRAG_COEFFICIENT = 1      # drag coefficient to use in the simulation

NEWTON_EPSILON = 0.001   # for newton method (implicit) - to stop iterating when we find a "almost root" smaller than this value
NEWTON_MAX_ITER = 100    # for newton method (implicit) - maximum number of iterations




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
        self.epsilon = NEWTON_EPSILON
        self.max_iter = NEWTON_MAX_ITER

        # Type of simulation
        self.is_reentry_sim = SIM_TO_RUN != PROJECTILE_SIM 
        self.orbit_or_escape_vel_sim = False
        self.is_horizontal_sim = False

        # Simulation options
        self.sim_round_earth = ROUND_EARTH
        self.sim_with_parachute = SIM_WITH_PARACHUTE 
        self.lift_perpendicular_to_velocity = LIFT_PERPENDICULAR_TO_VELOCITY
        self.max_angle_of_attack = MAX_ANGLE_OF_ATTACK

        # Plot options
        self.sim_name_for_image = SIM_NAME_FOR_IMAGE
        self.sims_to_show_in_plot_metrics = SIMS_TO_SHOW_IN_PLOT_METRICS
        self.save_plot_images = SAVE_PLOT_IMAGES

        # Initial conditions
        self.x_0 = 0
        self.altitude_0 = 130_000
        self.init_angles = np.negative(np.arange(start=0, stop=15.1, step=0.5))  # Angles in degrees --> we negate them because the path angle is measured down from the horizon
        self.init_velocities = np.arange(start=0, stop=15_001, step=300)    # Possible Initial velocities (m/s)

        # Capsule parameters
        self.capsule_mass = 12_000
        self.capsule_surface_area = 4 * np.pi
        self.capsule_drag_coefficient = CAPSULE_DRAG_COEFFICIENT
        self.capsule_lift_coefficient = CAPSULE_LIFT_COEFFICIENT

        # Parachute parameters
        self.parachute_surface_area = 301
        self.parachute_drag_coefficient = PARACHUTE_DRAG_COEFFICIENT
        self.parachute_max_open_altitude = 1_000
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
    if SIM_TO_RUN == REENTRY_SIM_NORMAL:
        return correct_exception_params(p)
    elif SIM_TO_RUN == REENTRY_SIM_CUSTOMIZED_PAIRS:
        p.init_angles = INIT_ANGLES
        p.init_velocities = INIT_VELOCITIES
        return correct_exception_params(p)
    elif SIM_TO_RUN == REENTRY_SIM_VERTICAL_MOV:
        p.x_0 = 100_000
        p.init_angles = [90]
        p.init_velocities = [0]
        return correct_exception_params(p)
    elif SIM_TO_RUN == REENTRY_SIM_HORIZONTAL_MOV:
        p.init_angles = [0]
        p.init_velocities = [1000]
        p.is_horizontal_sim = True
        return correct_exception_params(p)
    elif SIM_TO_RUN == REENTRY_SIM_ORBITAL_MOV:
        p.sim_max_time = 60 * 4
        p.init_angles = [0]
        p.init_velocities = [orbital_velocity(p.altitude_0)]
        p.orbit_or_escape_vel_sim = True
        return correct_exception_params(p)
    elif SIM_TO_RUN == REENTRY_SIM_ESCAPE_VEL_MOV:
        p.sim_max_time = 60 * 10
        p.init_angles = [-0.3]
        p.init_velocities = [ 10 * orbital_velocity(p.altitude_0)]
        p.orbit_or_escape_vel_sim = True
        return correct_exception_params(p)
    elif SIM_TO_RUN == PROJECTILE_SIM:
        p.altitude_0 = 0
        p.init_angles = [30, 45, 60]
        p.init_velocities = [300]
        return correct_exception_params(p)
    else: 
        raise Exception("Invalid SIM_TO_RUN value")
        

