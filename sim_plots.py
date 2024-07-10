from matplotlib import ticker
import matplotlib.pyplot as plt

from sim_common_fun import *
from sim_params import *
import os

############################################################################################################
#                         Options to save images of the plots
############################################################################################################

''' Options to save images of the plots '''
DOTS_PER_INCH = 300 # Quality of images (100 is good to see on screen... 300 is better to support some zoom) 
FOLDER_TO_SAVE_IMAGES = 'images'
os.makedirs(FOLDER_TO_SAVE_IMAGES, exist_ok=True)


def save_image(fig, p: Params, plot_name):
    ''' save the image of the plot with a unique name in the folder FOLDER_TO_SAVE_IMAGES. If the plot already exists, add a counter to the name.
        The image saved will be the plot viewed in the screen (so we can format it before closing and the image will be saved with the same format)'''
    if p.save_plot_images:
        image_name = f'{FOLDER_TO_SAVE_IMAGES}/{p.sim_name_for_image}_{plot_name}.png'
        counter = 1
        while os.path.exists(image_name):
            image_name = f'{FOLDER_TO_SAVE_IMAGES}/{p.sim_name_for_image}_{counter}_{plot_name}.png'
            counter += 1
        fig.savefig(image_name, dpi=DOTS_PER_INCH)





############################################################################################################
#                         plots for air density
############################################################################################################

def plot_air_density(f):
    '''plot the values of air density in function of the altitude'''
    x = np.linspace(-1000, 500000, 100000)
    y = f(x)

    plt.figure(figsize = (10,8))
    plt.plot(x, y, 'b')
    plt.plot(ALTITUDE, AIR_DENSITY, 'ro')
    plt.title('Cubic Spline Interpolation')
    plt.xlabel('altitude')
    plt.ylabel('air_density')
    plt.show()




############################################################################################################
#                               Plots for simulations
############################################################################################################

############################################################################################################
# All simulations results 

def plot_all_reentrys(success, accel, vel, before, after, p: Params):
    '''plot the parameter values for all reentry solutions'''
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    conditions = [success, accel, vel, before, after]
    labels = ['success', 'over accel', 'landed over speed', 'landed before', 'landed after']
    colors = ['lime', 'r', 'pink', 'blueviolet', 'b']

    for i, condition in enumerate(conditions):
        for angle, velocity in condition:
            ax.plot(velocity, round(angle,1), 'o', color=colors[i])
        ax.plot([], [], 'o', color=colors[i], label=labels[i])
    ax.legend()
    ax.set_xlabel('initial velocity (m/s)')
    ax.set_ylabel('angle (º)')
    title = 'Reentry Results for Pairs (angle,velocity); if failed in some condition, means that passed conditions above)' + ('' if len(success) > 0 else ' »»» No valid pairs found.')
    plt.title(title)
    plt.show()
    save_image(fig, p, 'conditions')



############################################################################################################
# Plot Simulation Metrics -> Function divided in step so we can plot while the simulation is running, instead of having to store all values and just plot at the end

MAX_ALTITUDE_TO_PLOT = ALTITUDE_0 + 70_000
MAX_HORIZONTAL_DISTANCE_TO_PLOT = MAX_HORIZONTAL_DISTANCE + 500_000
max_altitude = MAX_ALTITUDE_TO_PLOT
show_parachute_label = False
min_dist_label = max_dist_label = max_success_dist_label = None

def start_sims_metrics_plot(p: Params, total_sims_to_show): 
    fig, axs = plt.subplots(2, 3, figsize=(12, 10))
    show_random_plots = total_sims_to_show < (len(p.init_angles) * len(p.init_velocities))
    title = ('' if show_random_plots else f'{total_sims_to_show} ') + ('reentry' if p.is_reentry_sim else 'projectile') + ' simulations' + (f' (showing {total_sims_to_show} random plots)' if show_random_plots else '')
    fig.suptitle(title, fontsize=10)
    return fig, axs


def plot_metric(ax, x, x_label, y, y_label, init_values_lable, chute_open_idx, p: Params, is_altitude_plot=False):
    '''plot a metric'''
    global show_parachute_label, min_dist_label, max_dist_label, max_success_dist_label
    ax.plot(x, y, label=init_values_lable) 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # plot the parachute opening point (plot just the point and save the coordinates to plot the label later)
    if chute_open_idx > 0 and chute_open_idx < len(x):
        ax.scatter(x[chute_open_idx], y[chute_open_idx], color='green', marker='o', s=12)
        show_parachute_label = True 
    # plot the initial altitude circles (if it's a reentry simulation)
    if is_altitude_plot and p.is_reentry_sim:
        ax.scatter(x[0], y[0], color='blue', marker='o', s=10)        



def plot_sim_metrics(axs, sim_metrics, angle_0, v_0, is_reentry_sim, p: Params):
    sim = sim_metrics

    dist_label = f'Distance (m)'
    alt_label = f'Altitude (m)'      
    vel_label = f'Velocity (m/s)'
    acc_label = f'Acceleration (without G) (m/s^2)'
    time_label = f'Time (s)'

    init_values_lable = f'ang {angle_0:.1f}, vel {v_0:.0f},   '
        
    # trim the vectors to show only more 500km after the max landing distance
    idx_max = np.argmax(sim[PATH_X] > MAX_HORIZONTAL_DISTANCE_TO_PLOT)
    if idx_max != 0:
        for key, vector in sim.items():
            sim[key] = vector[:idx_max]
    
    # find the parachute opening point index    
    chute_open_idx = (np.argmax(sim[CHUTE_OPENING] > 0) - 1) if p.sim_with_parachute else -1


    ''' Path plot '''
    x_comp_label = f'x({p.x_0:.0f} / {max(sim[PATH_X]):.0f})'
    y_comp_label = ""  if is_reentry_sim else  f', y({min(sim[PATH_Y]):.0f} / {max(sim[PATH_Y]):.0f})'    
    
    # x=distance, y=altitude
    plot_metric(axs[0,0], sim[PATH_X], dist_label, sim[PATH_Y], alt_label, init_values_lable + x_comp_label + y_comp_label, chute_open_idx, p) 
    # save simulations landing distances to see if we plot the landing boudaries later
    global min_dist_label, max_dist_label, max_success_dist_label, max_altitude
    max_x = max(sim[PATH_X])  
    if min_dist_label is None and p.min_horizontal_distance <= max_x:
        min_dist_label = max_success_dist_label = p.min_horizontal_distance 
    if max_dist_label is None and p.max_horizontal_distance <= max_x: 
        max_dist_label = max_success_dist_label = p.max_horizontal_distance
    # save max altitude to limit the y axis in the altitude plots
    max_altitude = min(MAX_ALTITUDE_TO_PLOT, max(sim[PATH_Y]) * 1.1)


    # x=time, y=altitude
    plot_metric(axs[1,0], sim[TIMES], time_label, sim[PATH_Y], alt_label, init_values_lable, chute_open_idx, p, is_altitude_plot=True)


    ''' velocity plots '''
    # x=Altitude vs y=Velocity
    vel_comp_label = f'vel({min(sim[ABS_VELOCITIES]):.0f} / {max(sim[ABS_VELOCITIES]):.0f})'
    plot_metric(axs[0,1], sim[PATH_Y], alt_label, sim[ABS_VELOCITIES], vel_label, init_values_lable + vel_comp_label, chute_open_idx, p, is_altitude_plot=True)
                
    # x=Time vs y=Velocity
    plot_metric(axs[1,1], sim[TIMES], time_label, sim[ABS_VELOCITIES], vel_label, init_values_lable + vel_comp_label, chute_open_idx, p)


    ''' acceleration plots '''
    # trim the vectors to show only more 100 after max acceleration 
    idx_max = np.argmax(sim[ACCELERATIONS] > p.max_acceleration + 100 )
    if idx_max != 0:
        sim[PATH_Y] = sim[PATH_Y][:idx_max]
        sim[ACCELERATIONS] = sim[ACCELERATIONS][:idx_max]
        sim[TIMES] = sim[TIMES][:idx_max]
        
    # x=Altitude vs y=Acceleration
    acc_comp_label = f'acc({min(sim[ACCELERATIONS]):.0f} / {max(sim[ACCELERATIONS]):.0f})'
    plot_metric(axs[0,2], sim[PATH_Y], alt_label, sim[ACCELERATIONS], acc_label, init_values_lable + acc_comp_label, chute_open_idx, p)

    # x=Time vs y=Acceleration
    plot_metric(axs[1,2], sim[TIMES], time_label, sim[ACCELERATIONS], acc_label, init_values_lable + acc_comp_label, chute_open_idx, p)          




def end_sims_metrics_plot(fig, axs, p: Params): 
    global show_parachute_label, min_dist_label, max_dist_label, max_success_dist_label, max_altitude
    # plot extra information and labels in the plots  
    for ax in axs.flat:
        # plot parachute opening point label
        if show_parachute_label: 
            ax.scatter([], [], color='green', marker='o', s=10, label='Chute Open')
        # plots with acceleration on y axis 
        if ax in [axs[0,2], axs[1,2]]: 
            ax.axhline(y=p.max_acceleration, color='red', linestyle='--', label='max acceleration') # plot max acceleration line
        # plots with altitude on x axis 
        if p.is_reentry_sim and ax in [axs[0,1], axs[0,2]]: 
            if p.is_reentry_sim and not p.orbit_or_escape_vel_sim: # invert axis if reentry simulation 
                ax.invert_xaxis()
            ax.set_xlim(left=max_altitude) # limit the max altitude ploted
            ax.axvline(x=p.altitude_0, color='blue', linestyle='--', label='initial altitude') # plot initial altitude line
        # plots with altitude on y axis
        if p.is_reentry_sim and ax in [axs[0,0], axs[1,0]]: 
            ax.set_ylim(top=max_altitude) # limit the max altitude ploted
        # X_Y plot 
        if ax == axs[0,0]: 
            if min_dist_label is not None: # plot landing boundaries
                ax.axvline(x=min_dist_label, color='green', linestyle='--', label='min landing distance')
                ax.axvspan(min_dist_label, max_success_dist_label, color='lightgreen', alpha=0.1)
            if max_dist_label is not None:
                ax.axvline(x=max_dist_label, color='green', linestyle='--', label='max landing distance')
        # other plot configurations
        ax.legend(fontsize=6) 
        ax.tick_params(axis='both', which='major', labelsize=7)  # size of numbers on axis
        ax.ticklabel_format(useOffset=False, style='plain')      # avoid use of a base number shown on the side and scientific notation
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.grid()
    plt.tight_layout()
    plt.show()
    save_image(fig, p, 'metrics') 
    # reset global variables
    show_parachute_label = False
    min_dist_label = max_dist_label = max_success_dist_label = None


