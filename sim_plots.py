import matplotlib.pyplot as plt

from sim_common_fun import *
from sim_params import *


############################################################################################################

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



############################################################################################################

def plot_reentry_parameters(pairs):
    '''plot the parameter values that bound valid reentry solutions'''
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.96)
    for angle, velocity in pairs:
        ax.plot(velocity, -angle,'-o', color='b')
    ax.set_xlabel('initial velocity (m/s)')
    ax.set_ylabel('downward angle (º)')
    ax.legend()
    title = 'Reentry Parameters'
    if len(pairs) == 0:
        ax.legend('No valid reentry solutions found.')
    plt.title(title)
    plt.show()
    
def plot_all_reentrys(success, accel, vel, before, after):
    '''plot the parameter values for all reentry solutions'''
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.96)
    for angle, velocity in success:
        ax.plot(velocity, -angle,'-o', color='lime', label ='success' if angle == success[0][0] and velocity == success[0][1] else '')
    for angle, velocity in accel:
        ax.plot(velocity, -angle,'-o', color='r', label ='over accel' if angle == accel[0][0] and velocity == accel[0][1] else '')
    for angle, velocity in vel:
        ax.plot(velocity, -angle,'-o', color='blueviolet', label ='over speed' if angle == vel[0][0] and velocity == vel[0][1] else '')
    for angle, velocity in before:
        ax.plot(velocity, -angle,'-o', color='pink', label ='landed before' if angle == before[0][0] and velocity == before[0][1] else '')
    for angle, velocity in after:
        ax.plot(velocity, -angle,'-o', color='b', label ='landed after' if angle == after[0][0] and velocity == after[0][1] else '')
    ax.set_xlabel('initial velocity (m/s)')
    ax.set_ylabel('downward angle (º)')
    ax.legend()
    title = 'Reentry Parameters'
    if len(success) == 0:
        ax.legend('No valid reentry solutions found.')
    plt.title(title)
    plt.show()



############################################################################################################
# Plot Simulation Metrics -> Function divided in step so we can plot while the simulation is running, instead of having to store all values and just plot at the end

def start_sims_metrics_plot(is_reentry_sim, number_of_plots_to_show): 
    fig, axs = plt.subplots(2, 3, figsize=(12, 10))
    fig.suptitle(('Reentry' if is_reentry_sim else 'Projectile') + f' simulation. Showing {number_of_plots_to_show} random plots', fontsize=10)
    return axs

def plot_metric(ax, x, x_label, y, y_label, init_values_lable):
    '''plot a metric'''
    ax.plot(x, y, label=init_values_lable) 
    ax.legend(fontsize=6) 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis='both', which='major', labelsize=7)  # size of numbers on axis
    ax.grid()

def plot_sim_metrics(axs, sim_metrics, is_reentry_sim):
    sim = sim_metrics

    dist_label = f'Distance (m)'
    alt_label = f'Altitude (m)'      
    vel_label = f'Velocity (m/s)'
    acc_label = f'Acceleration (m/s^2)'
    time_label = f'Time (s)'

    if SHOW_DETAILS:
     
        init_values_lable = f'ang {sim[INIT_ANGLE]:.0f}, vel {sim[INIT_VELOCITY]:.0f},   '
        
        # Path (x=distance, y=altitude) 
        x_comp_label = f'x({min(sim[PATH_X]):.0f} / {max(sim[PATH_X]):.0f})'
        y_comp_label = ""  if is_reentry_sim else  f', y({min(sim[PATH_Y]):.0f} / {max(sim[PATH_Y]):.0f})'
        plot_metric(axs[0,0], sim[PATH_X], dist_label, sim[PATH_Y], alt_label, init_values_lable + x_comp_label + y_comp_label) 
        
        # x=Altitude vs y=Velocity
        vel_comp_label = f'vel({min(sim[VELOCITIES]):.0f} / {max(sim[VELOCITIES]):.0f})'
        plot_metric(axs[0,1], sim[PATH_Y], alt_label, sim[VELOCITIES], vel_label, init_values_lable + vel_comp_label)
                    
        # x=Time vs y=Velocity
        plot_metric(axs[1,1], sim[TIMES], time_label, sim[VELOCITIES], vel_label, init_values_lable + vel_comp_label)

        # x=Altitude vs y=Acceleration
        acc_comp_label = f'acc({min(sim[ACCELERATIONS]):.0f} / {max(sim[ACCELERATIONS]):.0f})'
        plot_metric(axs[0,2], sim[PATH_Y], alt_label, sim[ACCELERATIONS], acc_label, init_values_lable + acc_comp_label)

        # x=Time vs y=Acceleration
        plot_metric(axs[1,2], sim[TIMES], time_label, sim[ACCELERATIONS], acc_label, init_values_lable + acc_comp_label)          
        
def end_sims_metrics_plot():
    plt.tight_layout()
    plt.show()


