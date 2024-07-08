from matplotlib import ticker
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

def plot_success_reentries(pairs):
    '''plot the parameter values that bound valid reentry solutions'''
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.96)
    for angle, velocity in pairs:
        ax.plot(velocity, -angle,'-o', color='b')
    ax.set_xlabel('initial velocity (m/s)')
    ax.set_ylabel('downward angle (º)')
    ax.legend()
    title = 'Successful Reentry Parameters'
    if len(pairs) == 0:
        ax.legend('No valid reentry solutions found.')
    plt.title(title)
    plt.show()
    
def plot_all_reentrys(success, accel, vel, before, after):
    '''plot the parameter values for all reentry solutions'''
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    conditions = [success, accel, vel, before, after]
    labels = ['success', 'over accel', 'landed over speed', 'landed before', 'landed after']
    colors = ['lime', 'r', 'pink', 'blueviolet', 'b']

    for i, condition in enumerate(conditions):
        for angle, velocity in condition:
            ax.plot(velocity, -angle, 'o', color=colors[i])
        ax.plot([], [], 'o', color=colors[i], label=labels[i])
    ax.legend()
    ax.set_xlabel('initial velocity (m/s)')
    ax.set_ylabel('downward angle (º)')
    title = 'Reentry Results for Pairs: (angle,velocity)' + ('' if len(success) > 0 else ' »»» No valid pairs found.')
    plt.title(title)
    plt.show()



############################################################################################################
# Plot Simulation Metrics -> Function divided in step so we can plot while the simulation is running, instead of having to store all values and just plot at the end

show_parachute_label = False
min_dist_label = max_dist_label = max_success_dist_label = None

def start_sims_metrics_plot(p: Params, total_sims_to_show): 
    fig, axs = plt.subplots(2, 3, figsize=(12, 10))
    show_random_plots = total_sims_to_show < (len(p.init_angles) * len(p.init_velocities))
    title = ('' if show_random_plots else f'{total_sims_to_show} ') + ('reentry' if p.is_reentry_sim else 'projectile') + ' simulations' + (f' (showing {total_sims_to_show} random plots)' if show_random_plots else '')
    fig.suptitle(title, fontsize=10)
    return axs

def plot_metric(ax, x, x_label, y, y_label, init_values_lable, chute_open_idx, p: Params, is_x_y_plot = False, is_altitude_plot=False):
    '''plot a metric'''
    global show_parachute_label, min_dist_label, max_dist_label, max_success_dist_label
    # plot the metric (we trim the vector to show only more 500km after the max landing distance)
    idx_max = np.argmax(x > p.max_horizontal_distance + 500_000) if is_x_y_plot else len(x)
    x = x[:idx_max] if idx_max != 0 else x
    y = y[:idx_max] if idx_max != 0 else y
    ax.plot(x, y, label=init_values_lable) 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # plot the parachute opening point (plot just the point and save the coordinates to plot the label later)
    if chute_open_idx > 0 and chute_open_idx < len(x):
        ax.scatter(x[chute_open_idx], y[chute_open_idx], color='green', marker='o', s=12)
        show_parachute_label = True 
    # plot the initial altitude (if it's a reentry simulation)
    if is_altitude_plot and p.is_reentry_sim:
        ax.scatter(x[0], y[0], color='blue', marker='o', s=10)        
    # plot vertical lines of the landing boundaries, if they are within the x values (here we don't plot them, we just save the coordinates to plot them later)
    if is_x_y_plot:
        max_x = max(x)  
        if min_dist_label is None and p.min_horizontal_distance <= max_x:
            min_dist_label = max_success_dist_label = p.min_horizontal_distance 
        if max_dist_label is None and p.max_horizontal_distance <= max_x: 
            max_dist_label = max_success_dist_label = p.max_horizontal_distance


def plot_sim_metrics(axs, sim_metrics, is_reentry_sim, p: Params):
    sim = sim_metrics

    dist_label = f'Distance (m)'
    alt_label = f'Altitude (m)'      
    vel_label = f'Velocity (m/s)'
    y_vel_label = f'Vertical Velocity (Y) (m/s)'
    acc_label = f'Acceleration (without G) (m/s^2)'
    time_label = f'Time (s)'

    init_values_lable = f'ang {sim[INIT_ANGLE]:.1f}, vel {sim[INIT_VELOCITY]:.0f},   '
    chute_open_idx = (np.argmax(sim[CHUTE_OPENING] > 0) - 1) if p.sim_with_parachute else -1

    # Path (x=distance, y=altitude) 
    x_comp_label = f'x({min(sim[PATH_X]):.0f} / {max(sim[PATH_X]):.0f})'
    y_comp_label = ""  if is_reentry_sim else  f', y({min(sim[PATH_Y]):.0f} / {max(sim[PATH_Y]):.0f})'
    plot_metric(axs[0,0], sim[PATH_X], dist_label, sim[PATH_Y], alt_label, init_values_lable + x_comp_label + y_comp_label, chute_open_idx, p, is_x_y_plot=True) 
    
    # x=Altitude vs y=Y_Velocity
    yvel_comp_label = f'y_vel({min(sim[Y_VELOCITIES]):.0f} / {max(sim[Y_VELOCITIES]):.0f})'
    plot_metric(axs[1,0], sim[PATH_Y], alt_label, sim[Y_VELOCITIES], y_vel_label, init_values_lable + yvel_comp_label, chute_open_idx, p, is_altitude_plot=True)

    # x=Altitude vs y=Velocity
    vel_comp_label = f'vel({min(sim[ABS_VELOCITIES]):.0f} / {max(sim[ABS_VELOCITIES]):.0f})'
    plot_metric(axs[0,1], sim[PATH_Y], alt_label, sim[ABS_VELOCITIES], vel_label, init_values_lable + vel_comp_label, chute_open_idx, p, is_altitude_plot=True)
                
    # x=Time vs y=Velocity
    plot_metric(axs[1,1], sim[TIMES], time_label, sim[ABS_VELOCITIES], vel_label, init_values_lable + vel_comp_label, chute_open_idx, p)

    # x=Altitude vs y=Acceleration
    acc_comp_label = f'acc({min(sim[ACCELERATIONS]):.0f} / {max(sim[ACCELERATIONS]):.0f})'
    plot_metric(axs[0,2], sim[PATH_Y], alt_label, sim[ACCELERATIONS], acc_label, init_values_lable + acc_comp_label, chute_open_idx, p)

    # x=Time vs y=Acceleration
    plot_metric(axs[1,2], sim[TIMES], time_label, sim[ACCELERATIONS], acc_label, init_values_lable + acc_comp_label, chute_open_idx, p)          


def end_sims_metrics_plot(axs, p: Params): 
    global show_parachute_label, min_dist_label, max_dist_label, max_success_dist_label
    # plot extra information and labels in the plots  
    for ax in axs.flat:
        if show_parachute_label: # plot parachute opening point label
            ax.scatter([], [], color='green', marker='o', s=10, label='Chute Open')
        if ax in [axs[0,2], axs[1,2]]:# plot max acceleration line 
            ax.axhline(y=p.max_acceleration, color='red', linestyle='--', label='max acceleration')
        if ax in [axs[1,0], axs[0,1], axs[0,2]]: # plots with altitude, invert y axis if reentry simulation and plot initial altitude
            if p.is_reentry_sim and not p.orbit_or_escape_vel_sim:
                ax.invert_xaxis()
            ax.axvline(x=p.altitude_0, color='blue', linestyle='--', label='initial altitude')
        if ax == axs[0,0]: # plot landing boundaries in X_Y plot
            if min_dist_label is not None: 
                ax.axvline(x=min_dist_label, color='green', linestyle='--', label='min landing distance')
                ax.axvspan(min_dist_label, max_success_dist_label, color='lightgreen', alpha=0.1)
            if max_dist_label is not None:
                ax.axvline(x=max_dist_label, color='green', linestyle='--', label='max landing distance')
        ax.legend(fontsize=6) 
        ax.tick_params(axis='both', which='major', labelsize=7)  # size of numbers on axis
        ax.ticklabel_format(useOffset=False, style='plain')      # avoid use of a base number shown on the side and scientific notation
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.grid()
    plt.tight_layout()
    plt.show()
    # reset global variables
    show_parachute_label = False
    min_dist_label = max_dist_label = max_success_dist_label = None


