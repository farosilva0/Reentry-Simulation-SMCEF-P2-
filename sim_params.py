import numpy as np

from run_various_sims import *



######################################################################
#                   CHOOSE SIMULATION OPTIONS                        #
# sim_params are set according with the choosen options bellow       #
######################################################################


''' 1. Choose the simulation to run from options below '''
SIM_TO_RUN = 2
#------------------------
REENTRY_SIM = 1
PROJECTILE_SIM = 2          # simulation of a projectile being launched with different angles and velocities
#------------------------        


''' 2. Choose type of simulation from options below: '''
SIM_TYPE = 1
#------------------------
NORMAL_SIM = 1                  # we'll start simulation for several angles and velocities
NORMAL_SIM_BUT_LESS_PAIRS = 2  # we'll start simulation for less angles and velocities
HORIZONTAL_SIM = 3              # we'll start the simulation with some velocity and angle 0, and no forces, so the altitude will remain the same even in round earth
VERTICAL_SIM = 4                # we'll start the simulation without velocity, so with forces object will move vertically
                                # For vertical simulation, make sure LIFT = 0, because if not there will be horizontal movement; try with lift = 0 and = 1 to see the lift effect
ORBITAL_VEL_SIM = 5             # we'll start the simulation with the orbital velocity, so the object will keep the same altitude and will move horizontally
ESCAPE_VEL_SIM = 6              # we'll start the simulation with the escape velocity, so the object will keep the same altitude and will move horizontally
#------------------------



''' 3. Choose more options: '''
CONSTANT_GRAVITY = True            # if True we'll use constant values for gravity
CONSTANT_AIR_DENSITY = True        # if True we'll use constant values for air density

SIM_WITH_PARACHUTE = False          # if True we'll simulate the reentry with deployment of the parachutes after some conditions are met

ROUND_EARTH = False                  # if True we'll simulate the reentry in a round Earth

LIFT_PERPENDICULAR_TO_VELOCITY = False  # if False, all lift force will be added to y component regardless of velocity direction
                                        # if True, lift force will be perpendicular to velocity direction, and always pointing up
SHOW_DETAILS = True
# TODO: correr com SHOW_DETAILS = False e ver se os resultados são os mesmos, e se não são, ver o que está a ser mostrado que não devia ser mostrado


''' If you want different values than the default ones, choose here: '''
# TODO: meter isto a atualizar os params 
dt = 0.01                        # time steps (s)
SIM_MAX_TIME = 60 * 30            # max time for the simulation (s)
SIMS_TO_SHOW_IN_PLOT_METRICS = 10 # number of simulations to show in the plot metrics (we don't show all of them to not clutter the plot)

INIT_ANGLES = [45, -2, -4, -8]    # initial angles (degrees)
INIT_VELOCITIES = [100, 2_000, 4_000, 8_000] # initial velocities (m/s)

CAPSULE_DRAG_COEFFICIENT = 1.2      # drag coefficient to use in the simulation
CAPSULE_LIFT_COEFFICIENT = 1        # lift coefficient to use in the simulation

PARACHUTE_DRAG_COEFFICIENT = 1      # drag coefficient to use in the simulation
                                       


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
V, A, ACC_EARTH_ANGLE = 0, 1, 2



class Params:
    ''' inside a class to be easier to pass as a parameter to the functions'''
    def __init__(self):

        # Simulation details
        # TODO: confirmar precisões
        self.dt = 0.01
        self.sim_max_time = 60 * 30
        self.epsilon = 0.0001   # for newton method (implicit) - to stop iterating when we find a "almost root" smaller than this value
        self.max_iter = 1000    # for newton method (implicit) - maximum number of iterations
        self.show_details = True
        self.sims_to_show_in_plot_metrics = 10
        self.is_reentry_sim = SIM_TO_RUN == REENTRY_SIM
        

        # Initial conditions
        self.x_0 = 0
        self.altitude_0 = 130_000
        self.init_angles = np.negative(np.arange(start=0, stop=15.1, step=0.5))  # Angles in degrees --> we negate them because the path angle is measured down from the horizon
        self.init_velocities = np.arange(start=0, stop=15_001, step=300)    # Possible Initial velocities (m/s)

        # Capsule parameters
        self.capsule_mass = 12_000
        self.capsule_surface_area = 4 * np.pi
        self.capsule_drag_coefficient = 1.2
        self.capsule_lift_coefficient = 1

        # Parachute parameters
        self.parachute_surface_area = 301
        self.parachute_drag_coefficient = 1
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
               f"show_details = {self.show_details}\n" \
               f"sims_to_show_in_plot_metrics = {self.sims_to_show_in_plot_metrics}\n"


def correct_exception_params(p: Params):
    #TODO: verificar as formulas e ver se dá pra não dividir pela velocidade 
    for i, vel in enumerate(p.init_velocities):
        if vel == 0:
            p.init_velocities[i] = 0.000001
    if p.capsule_mass == 0:
        p.capsule_mass = 0.000001
    return p


def get_params():
    p = Params()
    if SIM_TO_RUN == REENTRY_SIM:
        if SIM_TYPE == NORMAL_SIM:
            return correct_exception_params(p)
        elif SIM_TYPE == NORMAL_SIM_BUT_LESS_PAIRS:
            p.init_angles = [0, -8, -12, -16, -30, -50]
            p.init_velocities = [0, 2_000, 4_000, 8_000, 15_000]
            return correct_exception_params(p)
    if SIM_TO_RUN == PROJECTILE_SIM:
        if SIM_TYPE == NORMAL_SIM:
            p.altitude_0 = 0
            p.init_angles = [30, 45, 60]
            p.init_velocities = [100]
            p.capsule_mass = 1
            p.capsule_surface_area = 1
            p.capsule_drag_coefficient = 0
            p.capsule_lift_coefficient = 0
            p.parachute_drag_coefficient = 0
            return correct_exception_params(p)
        

