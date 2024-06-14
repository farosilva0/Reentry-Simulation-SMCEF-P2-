import numpy as np
import matplotlib.pyplot as plt



# SIMULATION PARAMETERS
dt = 0.01            # time steps (s) 
altitude_0 = 130_000   # Initial altitude (m)
angle_0 = 0       # Entry angle (degrees)
v_0 = 7_800         # Initial velocity (m/s)

# Capsule parameters
CAPSULE_MASS = 12_000    # Mass of the capsule (kg)
CAPSULE_SURFACE_AREA = 4 * np.pi  # surface considered for drag coefficient (m^2)
CAPSULE_DRAG_COEFFICIENT = 1.2  # drag coefficient
CAPSULE_LIFT_COEFFICIENT = 1.0  # lift coefficient

# Physical constants
G_M = 6.67430e-11 * 5.972e24    # G (Gravitational constant) * Earth mass, m^3 kg^-1 s^-2
RADIUS_EARTH = 6.371e6          # Earth radius (m)



def getAirDensity(y):
    '''returns the air density at a given altitude'''
    # Model of the air density variation with altitude
    rho0 = 1.225  # air density at sea level (kg/m^3)
    h_scale = 8500.0  # scale height (m)
    altitude = y - RADIUS_EARTH
    return rho0 * np.exp(-altitude / h_scale)



def drag_acceleration(vx, vy, v, y, A = CAPSULE_SURFACE_AREA, Cd = CAPSULE_DRAG_COEFFICIENT):
    '''calculates the drag acceleration on the object, given its velocity and altitude.
        By default, the drag force is calculated for the capsule.
        Drag force always opposes the movement direction, so it is negative.'''
    Fd = -0.5 * Cd * A * getAirDensity(y) * v**2  #
    ax = Fd * vx / CAPSULE_MASS
    ay = Fd * vy / CAPSULE_MASS
    return ax, ay


def lift_acceleration(vx, vy, v, y, A = CAPSULE_SURFACE_AREA, Cl = CAPSULE_LIFT_COEFFICIENT):
    '''calculates the lift acceleration on the object, given its velocity and altitude.
        By default, the lift force is calculated for the capsule.
        Lift force is perpendicular to the movement direction, so it is calculated using the velocity components.'''
    Fl = 0.5 * Cl * A * getAirDensity(y) * v**2
    ax = Fl * vy / (CAPSULE_MASS * v)   
    ay = -Fl * vx / (CAPSULE_MASS * v)  # Lift force oposes the fall (y component)
    return ax, ay


def gravity_acceleration(x, y):
    '''calculates the acceleration due to gravity at a given position (x, y)
        returns the acceleration components gx and gy. '''
    r = np.sqrt(x**2 + y**2)    # distance from the center of the Earth
    g = - G_M / r**2            # gravity acceleration pointing to the center of the Earth (negative because it goes down) 
    gx = g * (x / r)           
    gy = g * (y / r)
    return gx, gy


def total_acceleration(x, y, vx, vy):
    '''calculates the total acceleration on the capsule'''
    v = np.sqrt(vx**2 + vy**2) # velocity in the movement's direction
    dx, dy = drag_acceleration(vx, vy, v, y)
    lx, ly = lift_acceleration(vx, vy, v, y)
    # TODO: calcular as 2 acelarações em conjunto?? ou tem de ser separado??
    gx, gy = gravity_acceleration(x, y)
    # TODO: somar as acelerações todas
    return gx, gy


def run_entry_simulation(altitude_0, angle_0, v_0):
    '''runs a simulation of the capsule reentry'''

    # initial velocity
    angle_radians = np.radians(angle_0)
    vx_0 = v_0 * np.cos(angle_radians)
    vy_0 = v_0 * np.sin(angle_radians)

    # accumulator variables for the simulation
    x = 0                       # x position
    y = RADIUS_EARTH + altitude_0  # y position
    vx = vx_0                   # x velocity
    vy = vy_0                   # y velocity    
    
    path_x = [x]
    path_y = [y - RADIUS_EARTH]
    
    while y > RADIUS_EARTH:
        ''' reentry simulation using Euler's method'''
        '''TODO: Runge-Kutta methods for better accuracy'''
        '''TODO: analitic methods...'''
        # total acceleration
        ax, ay = total_acceleration(x, y, vx, vy)
        
        # velocity 
        vx += ax * dt
        vy += ay * dt
        
        # positions
        x += vx * dt
        y += vy * dt
        
        # store positions
        path_x.append(x)
        path_y.append(y - RADIUS_EARTH)  # Altitude em relação ao nível do mar
               
    return path_x, path_y


def plot_path(path_x, path_y):
    '''plot the path of the capsule'''
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(path_x, path_y, label='Capsule Path')
    ax.set_xlabel('x distance (m)')
    ax.set_ylabel('y altitude (m)')
    ax.legend()
    plt.show()

def main():
    path_x, path_y = run_entry_simulation(altitude_0, angle_0, v_0)
    plot_path(path_x, path_y)

if __name__ == "__main__":
    main()
