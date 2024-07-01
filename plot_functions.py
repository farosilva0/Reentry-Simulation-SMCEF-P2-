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





def plot_metric(ax, x, x_label, y, y_label, init_values_lable, invert_x_values = False, invert_y_values = False):
    '''plot a metric'''
    ax.plot(x, y, label=init_values_lable) 
    ax.legend(fontsize=7) 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if invert_x_values:
        ax.invert_xaxis()
    if invert_y_values:
        ax.invert_yaxis()
    ax.grid()


def plot_sims_metrics(tot_sims_metrics, reentry_sim):
    '''plot path, and if SHOW_DETAILS is True, plot also acceleration and velocity. '''   
    dist_label = f'Distance (m)'
    alt_label = f'Altitude (m)'      
    vel_label = f'Velocity (m/s)'
    acc_label = f'Acceleration (m/s^2)'
    time_label = f'Time (s)'


    if SHOW_DETAILS:
        fig, axs = plt.subplots(2, 3, figsize=(12, 10))
        for sim in tot_sims_metrics: 

            init_values_lable = f'ang:{sim[INIT_ANGLE]:.0f}, vel:{sim[INIT_VELOCITY]:.0f}'
            
            # Path (x=distance, y=altitude) 
            path_label = f', max_x:{max(sim[PATH_X]):.0f}, max_y:{max(sim[PATH_Y]):.0f}'
            plot_metric(axs[0,0], sim[PATH_X], dist_label, sim[PATH_Y], alt_label, init_values_lable + path_label) 

            # x=Altitude vs y=Velocity
            vel_label = f', max_v:{max(sim[VELOCITIES]):.0f}, min_v:{min(sim[VELOCITIES]):.0f}'
            plot_metric(axs[0,1], sim[PATH_Y], alt_label, sim[VELOCITIES], vel_label, init_values_lable + vel_label, invert_x_values = reentry_sim, invert_y_values = True)
                        
            # x=Time vs y=Velocity
            plot_metric(axs[1,1], sim[TIMES], time_label, sim[VELOCITIES], vel_label, init_values_lable + vel_label, invert_x_values = False, invert_y_values = True)

            # x=Altitude vs y=Acceleration
            acc_label = f', max_a:{max(sim[ACCELERATIONS]):.0f}, min_a:{min(sim[ACCELERATIONS]):.0f}'
            plot_metric(axs[0,2], sim[PATH_Y], alt_label, sim[ACCELERATIONS], acc_label, init_values_lable + acc_label, invert_x_values = reentry_sim, invert_y_values = True)

            # x=Time vs y=Acceleration
            plot_metric(axs[1,2], sim[TIMES], time_label, sim[ACCELERATIONS], acc_label, init_values_lable + acc_label, invert_x_values = False, invert_y_values = True)
            
        plt.tight_layout()
        plt.show()

    else: # NOT SHOW_DETAILS - only plot the path
        # Path (x=distance, y=altitude) 
        init_values_lable = f'angle: {sim[INIT_ANGLE]:.1f}, velocity: {sim[INIT_VELOCITY]:.1f}'
        plot_metric(axs[0,0], sim[PATH_X], dist_label, sim[PATH_Y], alt_label, init_values_lable) 


