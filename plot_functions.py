from reentry_sim import *
import matplotlib.pyplot as plt


# Simulation Metrics Names
INIT_ANGLE = 'init_angle'
INIT_VELOCITY = 'init_velocity' 
TIMES = 'times'
PATH_X = 'path_x'
PATH_Y = 'path_y'
VELOCITIES = 'velocities'
ACCELERATIONS = 'accelerations'



def plot_air_density(f):
    '''plot the values of air density in function of the altitude'''
    x = np.linspace(-1000, 500000, 1000000)
    y = f(x)

    plt.figure(figsize = (10,8))
    plt.plot(x, y, 'b')
    plt.plot(ALTITUDE, AIR_DENSITY, 'ro')
    plt.title('Cubic Spline Interpolation')
    plt.xlabel('altitude')
    plt.ylabel('air_density')
    plt.show()


def plot_reentry_parameters(pairs):
    '''plot the parameter values that bound valid reentry solutions'''
    if len(pairs) == 0:
        print('No valid reentry solutions found.')
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for angle, velocity in pairs:
        ax.plot(velocity, -angle,'-o', label=f'angle: {angle}')
    ax.set_xlabel('initial velocity (m/s)')
    ax.set_ylabel('downward angle (º)')
    ax.legend()
    plt.show()



def plot_sims_metrics(tot_sims_metrics):
    '''plot path, and if SHOW_DETAILS is True, plot also acceleration and velocity. '''   
    if SHOW_DETAILS:
        fig, axs = plt.subplots(2, 3, figsize=(12, 10))
        for sim in tot_sims_metrics:
            times = sim[TIMES]
            x_km = np.array(sim[PATH_X]) / 1000.0  # Convert distances to kilometers
            y_km = np.array(sim[PATH_Y]) / 1000.0  # Convert altitudes to kilometers
                
            # Path (x * y)
            label = f'angle: {sim[INIT_ANGLE]:.1f}, velocity: {sim[INIT_VELOCITY]:.1f}'
            ax = axs[0,0]
            if RUN_OTHER_SIM:
                ax.plot(sim[PATH_X], sim[PATH_Y], label=label)
                ax.set_xlabel('x distance (m)')
                ax.set_ylabel('y altitude (m)')
            else:
                ax.plot(x_km, y_km, label=label)
                ax.set_xlabel('x distance (km)')
                ax.set_ylabel('y altitude (km)')
            ax.legend()
            ax.grid()

            # Velocity vs Altitude
            ax = axs[0, 1]
            ax.plot(y_km, sim[VELOCITIES])
            ax.set_xlabel('Altitude (km)')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_xlim(max(y_km), min(y_km)) # Invert x-axis to see from left to right
            ax.grid()

            # Velocity vs Time
            ax = axs[1, 1]
            ax.plot(times, sim[VELOCITIES])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (m/s)')
            ax.grid()

            # Acceleration vs Altitude
            ax = axs[0,2]
            ax.plot(y_km, sim[ACCELERATIONS])
            ax.set_xlabel('Altitude (km)')
            ax.set_ylabel('Acceleration (m/s^2)')
            ax.set_xlim(max(y_km), min(y_km)) # Invert x-axis to see from left to right
            ax.grid()

            # Acceleration vs Time
            ax = axs[1,2]
            ax.plot(times, sim[ACCELERATIONS])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Acceleration (m/s^2)')
            ax.grid()


        plt.tight_layout()
        plt.show()

    else: # NOT SHOW_DETAILS - only plot the path
        fig, ax = plt.subplots(figsize=(12, 8))
        for sim in tot_sims_metrics:
            path_x = sim[PATH_X]
            path_y = sim[PATH_Y]
            label = f'angle: {sim[INIT_ANGLE]:.1f}, velocity: {sim[INIT_VELOCITY]:.1f}'
            ax.plot(path_x, path_y, label=label)
        ax.set_xlabel('x distance (m)')
        ax.set_ylabel('y altitude (m)')
        ax.legend()
        plt.show()




    # # Acceleration vs Time
    # plt.figure()
    # plt.plot(times, accelerations)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Acceleration (m/s^2)')
    # plt.title('Acceleration vs Time')
    # plt.grid()
    # plt.show()

def plot_aceleration_vs_altitude(accelerations, altitudes):
    # Acceleration vs Altitude
    plt.figure()
    plt.plot(altitudes, accelerations)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Acceleration vs Altitude')
    plt.grid()
    plt.show()

def plot_velocity_vs_time(velocities, times):
    # Velocity vs Time
    plt.figure()
    plt.plot(times, velocities)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.grid()
    plt.show()

def plot_velocity_vs_altitude(velocities, altitudes):
    # Velocity vs Altitude
    plt.figure()
    plt.plot(altitudes, velocities)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Altitude')
    plt.grid()
    plt.show()
