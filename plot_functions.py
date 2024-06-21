from reentry_sim import *
import matplotlib.pyplot as plt



def plot_path(path_x, path_y):
    '''plot the path of the capsule'''
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(path_x, path_y, label='Capsule Path')
    ax.set_xlabel('x distance (m)')
    ax.set_ylabel('y altitude (m)')
    ax.legend()
    plt.show()


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


def plot_reentry_parameters(angles):
    '''plot the parameter values that bound valid reentry solutions'''

    fig, ax = plt.subplots(figsize=(12, 8))
    for angle, velocities in angles:
        ax.plot(velocities, angle,'-o', label=f'angle: {angle[0]}')
    ax.set_xlabel('initial velocity (m/s)')
    ax.set_ylabel('downward angle (º)')
    ax.legend()
    plt.show()