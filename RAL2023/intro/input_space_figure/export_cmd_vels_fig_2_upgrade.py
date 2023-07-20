# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from models.kinematic.ideal_diff_drive import Ideal_diff_drive

# matplotlib.get_backend()
# matplotlib.use('MacOSX')
# matplotlib.use('QtAgg')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Prepare data                                                                                         ::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# === Setup data from snow_train.csv =======================================================================
df = pd.read_csv("snow_train.csv", dtype={
        '':               int, 'ros_time': np.int64, 'joy_switch': str, 'icp_index': int, 'calib_state': str,
        'estop':          str, 'cmd_left_vel': float, 'meas_left_vel': float, 'cmd_right_vel': float,
        'meas_right_vel': float, 'cmd_vel_x': float, 'cmd_vel_omega': float, 'icp_pos_x': float, 'icp_pos_y': float,
        'icp_pos_z':      float, 'icp_quat_x': float, 'icp_quat_y': float, 'icp_quat_z': float, 'icp_quat_w': float,
        'voltage_left':   float, 'current_left': float, 'voltage_right': float, 'current_right': float, 'imu_x': float,
        'imu_y':          float, 'imu_z': float
        })
time = (df["ros_time"][1:] - df["ros_time"][1:].iloc[0]) / 1e9
cmd_left_vel = df["cmd_left_vel"][1:]
cmd_right_vel = df["cmd_right_vel"][1:]
model = Ideal_diff_drive(0.3, 1.1652, 0.05)
cmd_vel_x, cmd_vel_omega = model.compute_body_vel(np.vstack([cmd_left_vel, cmd_right_vel]))

# # === Figure: Human-commanded speeds ======================================================================
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].plot(time, cmd_vel_x, color="C0")
# ax[0].set_xlabel("Time (s)")
# ax[0].set_ylabel("Linear speed (m/s)")
# ax[1].plot(time, cmd_vel_omega, color="C1")
# ax[1].set_xlabel("Time (s)")
# ax[1].set_ylabel("Angular speed (rad/s)")
# fig.suptitle("Human-commanded speeds")
# plt.show()


# === Setup data from ral2023_dataset ======================================================================
train_dataset = pd.read_pickle("../.././../data/ral2023_dataset/warthog_wheels/gravel_1/acceleration_dataset.pkl")
train_dataset_2 = pd.read_pickle(
        "../.././../data/ral2023_dataset/warthog_tracks/grand-axe_crusted-snow/acceleration_dataset.pkl")

# train_dataset = pd.read_pickle("../.././../data/ral2023_dataset/husky/grand_salon_tile_inflated
# /acceleration_dataset.pkl")
# train_dataset_2 = pd.read_pickle("../.././../data/ral2023_dataset/husky/boreal_snow/acceleration_dataset.pkl")


# ... extract cmd_body_vel arrays (input arrays) ...........................................................
idd_body_vel_x_str_list = []
idd_body_vel_y_str_list = []
idd_body_vel_yaw_str_list = []
for i in range(0, 40):
    str_idd_vel_x_i = 'idd_vel_x_' + str(i)
    str_idd_vel_y_i = 'idd_vel_y_' + str(i)
    str_idd_vel_yaw_i = 'idd_vel_yaw_' + str(i)
    idd_body_vel_x_str_list.append(str_idd_vel_x_i)
    idd_body_vel_y_str_list.append(str_idd_vel_y_i)
    idd_body_vel_yaw_str_list.append(str_idd_vel_yaw_i)
idd_body_vel_x_array = train_dataset[idd_body_vel_x_str_list].to_numpy()
idd_body_vel_y_array = train_dataset[idd_body_vel_y_str_list].to_numpy()
idd_body_vel_yaw_array = train_dataset[idd_body_vel_yaw_str_list].to_numpy()

idd_vel_x_array_tracks = train_dataset_2[idd_body_vel_x_str_list].to_numpy()
idd_vel_y_array_tracks = train_dataset_2[idd_body_vel_y_str_list].to_numpy()
idd_vel_yaw_array_tracks = train_dataset_2[idd_body_vel_yaw_str_list].to_numpy()

x_train = np.column_stack((idd_body_vel_x_array.flatten(), idd_body_vel_yaw_array.flatten()))

wheel_radius = 0.3
baseline = 1.1652
ideal_diff_drive = Ideal_diff_drive(wheel_radius, baseline, dt=0.05)
encoder_left_str_list = []
encoder_right_str_list = []
for i in range(0, 40):
    str_encoder_left_i = 'left_wheel_vel_' + str(i)
    str_encoder_right_i = 'right_wheel_vel_' + str(i)
    encoder_left_str_list.append(str_encoder_left_i)
    encoder_right_str_list.append(str_encoder_right_i)
encoder_left_vels = train_dataset[encoder_left_str_list].to_numpy()
encoder_right_vels = train_dataset[encoder_right_str_list].to_numpy()
encoder_body_vel_x = np.zeros((encoder_left_vels.shape[0], 40))
encoder_body_vel_yaw = np.zeros((encoder_left_vels.shape[0], 40))
for i in range(0, encoder_left_vels.shape[0]):
    for j in range(0, 40):
        body_vel = ideal_diff_drive.compute_body_vel(np.array([encoder_left_vels[i, j], encoder_right_vels[i, j]]))
        encoder_body_vel_x[i, j] = body_vel[0]
        encoder_body_vel_yaw[i, j] = body_vel[1]

# ... extract icp vels .....................................................................................
str_icp_vel_x_list = []
str_icp_vel_y_list = []
str_icp_vel_yaw_list = []
for i in range(0, 40):
    str_icp_vel_x_i = 'icp_vel_x_' + str(i)
    str_icp_vel_y_i = 'icp_vel_y_' + str(i)
    str_icp_vel_yaw_i = 'icp_vel_yaw_' + str(i)
    str_icp_vel_x_list.append(str_icp_vel_x_i)
    str_icp_vel_y_list.append(str_icp_vel_y_i)
    str_icp_vel_yaw_list.append(str_icp_vel_yaw_i)
icp_vel_x_array = train_dataset[str_icp_vel_x_list].to_numpy()
icp_vel_y_array = train_dataset[str_icp_vel_y_list].to_numpy()
icp_vel_yaw_array = train_dataset[str_icp_vel_yaw_list].to_numpy()

icp_vel_x_array_tracks = train_dataset_2[str_icp_vel_x_list].to_numpy()
icp_vel_y_array_tracks = train_dataset_2[str_icp_vel_y_list].to_numpy()
icp_vel_yaw_array_tracks = train_dataset_2[str_icp_vel_yaw_list].to_numpy()

# ... extract body_vel_distruptions arrays (output arrays) .................................................
str_body_vel_disturption_x_list = []
str_body_vel_disturption_y_list = []
str_body_vel_disturption_yaw_list = []
for i in range(0, 40):
    str_body_vel_disturption_x_i = 'body_vel_disturption_x_' + str(i)
    str_body_vel_disturption_y_i = 'body_vel_disturption_y_' + str(i)
    str_body_vel_disturption_yaw_i = 'body_vel_disturption_yaw_' + str(i)
    str_body_vel_disturption_x_list.append(str_body_vel_disturption_x_i)
    str_body_vel_disturption_y_list.append(str_body_vel_disturption_y_i)
    str_body_vel_disturption_yaw_list.append(str_body_vel_disturption_yaw_i)

body_vel_disturption_x_array = train_dataset[str_body_vel_disturption_x_list].to_numpy()
body_vel_disturption_y_array = train_dataset[str_body_vel_disturption_y_list].to_numpy()
body_vel_disturption_yaw_array = train_dataset[str_body_vel_disturption_yaw_list].to_numpy()

y_train_longitudinal_slip = body_vel_disturption_x_array.flatten()
y_train_lateral_slip = body_vel_disturption_y_array.flatten()
y_train_angular_slip = body_vel_disturption_yaw_array.flatten()

# ... compute mean body vel disturbance for each steady-state window .......................................
n_windows = len(train_dataset)

steady_state_mask = train_dataset['steady_state_mask'].to_numpy() == True
steady_state_mask_tracks = train_dataset_2['steady_state_mask'].to_numpy() == True

steady_state_idd_body_vel_x = idd_body_vel_x_array[steady_state_mask]
steady_state_idd_body_vel_y = idd_body_vel_y_array[steady_state_mask]
steady_state_idd_body_vel_yaw = idd_body_vel_yaw_array[steady_state_mask]

steady_state_idd_body_vel_x_tracks = idd_vel_x_array_tracks[steady_state_mask_tracks]
steady_state_idd_body_vel_y_tracks = idd_vel_y_array_tracks[steady_state_mask_tracks]
steady_state_idd_body_vel_yaw_tracks = idd_vel_yaw_array_tracks[steady_state_mask_tracks]

steady_state_encoder_body_vel_x = encoder_body_vel_x[steady_state_mask]
steady_state_encoder_body_vel_yaw = encoder_body_vel_yaw[steady_state_mask]

steady_state_icp_body_vel_x = icp_vel_x_array[steady_state_mask]
steady_state_icp_body_vel_y = icp_vel_y_array[steady_state_mask]
steady_state_icp_body_vel_yaw = icp_vel_yaw_array[steady_state_mask]

steady_state_icp_body_vel_x_tracks = icp_vel_x_array_tracks[steady_state_mask_tracks]
steady_state_icp_body_vel_y_tracks = icp_vel_y_array_tracks[steady_state_mask_tracks]
steady_state_icp_body_vel_yaw_tracks = icp_vel_yaw_array_tracks[steady_state_mask_tracks]

steady_state_body_vel_disturption_x = body_vel_disturption_x_array[steady_state_mask]
steady_state_body_vel_disturption_y = body_vel_disturption_y_array[steady_state_mask]
steady_state_body_vel_disturption_yaw = body_vel_disturption_yaw_array[steady_state_mask]

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Figure 1: Complete input-space calibration                                                           ::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
mpl.rcParams['lines.dashed_pattern'] = [2, 2]
mpl.rcParams['lines.linewidth'] = 1.0


width = 3.402
# height = width / 1.3
height = width / 1.1
plt.close('all')

# fig, ax = plt.subplots(1, 1, dpi=150)     # Original
fig, ax = plt.subplots(1, 1, dpi=300)
fig.set_size_inches(width, height)
# fig.subplots_adjust(left=.1, bottom=0.07, right=.995, top=.98)
# fig.subplots_adjust(top=0.99, bottom=0.32, right=.99, left=0.13)
fig.subplots_adjust(top=0.99, bottom=0.42, right=.99, left=0.13)


# ... .Tick color setup ....................................................................................
tick_color= 'darkgray'
plt.setp(ax.get_xticklabels(), color=tick_color)
plt.setp(ax.get_yticklabels(), color=tick_color)
ax.tick_params(axis='both', which='both', color=tick_color)


# ... Grid .................................................................................................
# ax.grid(which='major', color='gray', linestyle='--', alpha=0.15)
# ax.grid(which='minor', color='gray', linestyle='--', alpha=0.5)
# ax.grid(which='major',
#         # color='gray',
#         color='lightgray',
#         linestyle=(24, (24, 8)), linewidth=0.25,
#         # alpha=0.75
#         alpha=0.7
#         )

# --- Plot configuration -----------------------------------------------------------------------------------

# ... Global config.........................................................................................
alpha = 0.3

num_points = 100
# line_width = 1                  # original fig 1
line_width = 0.75
# line_width = 0.25

# ... Dots config ..........................................................................................
# dots_size = 1                 # original fig 1
# alpha_dots = 0.01             # original fig 1
# dots_edgecolors = 'face'      # original fig 1
# dots_size = 0.75
# alpha_dots = 0.4
# dots_size = 0.45
dots_size = 1.9
# alpha_dots = 0.5
# alpha_dots = 0.02
alpha_dots = 0.0
# dots_size = 0.35
# alpha_dots = 0.4
dots_edgecolors = 'none'

# ... Naive config..........................................................................................
alpha_naive = 0.11

# naive_edge_color = 'grey'
# naive_patern = "------"  # Version 1
naive_patern = ""

naive_edge_color = 'darkgray'
# naive_edge = (0, (10, 10))
naive_edge = (0, (16, 8, 6, 8))
naive_line_width = line_width * 1.6

# ... Human config .........................................................................................
# alpha_red = 0.15
# red_patern = "||||||"
# red_patern = "||||||||"
# red_patern = "/////////"

# alpha_red = 0.25
# red_patern = "++++"             # Version 1

# alpha_red = 0.11
red_patern = ""

alpha_red = 0.16
# human_edge = (0, (6, 6))
# human_edge = (0, (10, 10))
human_edge = naive_edge
human_line_width = naive_line_width

# ... Snow .................................................................................................
# color_snow = 'C0'
# alpha_snow = alpha
# alpha_dots_snow = alpha_dots

# color_snow = 'gray'
# alpha_snow = 0.2
# alpha_dots_snow = 0.025

color_snow = 'C0'
alpha_snow = 0.
alpha_dots_snow = 0.

# color_snow = 'white'
# alpha_snow = 0.4
# alpha_dots_snow = 0.1

# ... Plot boundaries ......................................................................................
minimum_linear_vel_positive = 0
minimum_linear_vel_negative = 0
minimum_angular_vel_positive = 0
minimum_angular_vel_negative = 0
maximum_linear_vel_positive = 5
maximum_linear_vel_negative = -5
maximum_angular_vel_positive = 4
maximum_angular_vel_negative = -4
gravel_maximum_linear_vel_positive = 4
gravel_maximum_linear_vel_negative = -4
gravel_maximum_angular_vel_positive = 2.5
gravel_maximum_angular_vel_negative = -2.5
snow_maximum_linear_vel_positive = 2.5
snow_maximum_linear_vel_negative = -2.5
snow_maximum_angular_vel_positive = 1
snow_maximum_angular_vel_negative = -1
human_maximum_linear_vel_positive = 5
human_maximum_linear_vel_negative = 0
human_maximum_angular_vel_positive = 2
human_maximum_angular_vel_negative = -2

x_min = -4
x_max = 4
y_min = -5
y_max = 5

# # === Naive ================================================================================================
# cmd_angular_vel_linspace = np.linspace(x_min, x_max, num_points)
# cmd_linear_max_vel_linspace = np.linspace(y_max, y_max, num_points)
# cmd_linear_min_vel_linspace = np.linspace(y_min, y_min, num_points)
#
# # ... plot initial input space .............................................................................
# ax.plot(cmd_angular_vel_linspace, cmd_linear_min_vel_linspace,
#         color=naive_edge_color,
#         lw=naive_line_width, label='Uncharacterized',
#         linestyle=naive_edge,
#         )
# ax.plot(cmd_angular_vel_linspace, cmd_linear_max_vel_linspace,
#         color=naive_edge_color,
#         lw=naive_line_width,
#         linestyle=naive_edge,
#         )
# ax.vlines(x_min, y_min, y_max,
#           color=naive_edge_color,
#           lw=naive_line_width,
#           linestyle=naive_edge,
#           )
# ax.vlines(x_max, y_min, y_max,
#           color=naive_edge_color,
#           lw=naive_line_width,
#           linestyle=naive_edge,
#           )
# ax.fill_between(cmd_angular_vel_linspace, cmd_linear_max_vel_linspace, y2=cmd_linear_min_vel_linspace,
#                     lw=0,
#                 alpha=alpha_naive,
#                 color='grey',
#                 hatch=naive_patern,
#                 )
#
# # === human ================================================================================================
# human_angular_vel_linspace_negative = np.linspace(human_maximum_angular_vel_negative, 0, int(num_points /
# 2)).flatten()
# human_angular_vel_linspace_positive = np.linspace(0, human_maximum_angular_vel_positive, int(num_points /
# 2)).flatten()
# human_angular_vel_linspace_all = np.linspace(human_maximum_angular_vel_negative, human_maximum_angular_vel_positive,
#                                              int(num_points / 2)).flatten()
# human_q1_vel_linspace = np.linspace(human_maximum_linear_vel_positive, 0, int(num_points / 2)).flatten()
# human_q2_vel_linspace = np.linspace(0, human_maximum_linear_vel_positive, int(num_points / 2)).flatten()
#
# # ... plot human ...........................................................................................
# q1_human_input_space = ax.plot(human_angular_vel_linspace_negative, human_q1_vel_linspace, color='C3',
#                                lw=human_line_width,
#                                label='Characterized',
#                                linestyle=human_edge,
#                                )
# q2_human_input_space = ax.plot(human_angular_vel_linspace_positive, human_q2_vel_linspace, color='C3',
#                                lw=human_line_width,
#                                linestyle=human_edge,
#                                )
# q3_human_input_space = ax.plot(human_angular_vel_linspace_all,
#                                np.full(int(num_points / 2), human_maximum_linear_vel_positive), color='C3',
#                                lw=human_line_width,
#                                linestyle=human_edge,
#                                )
# ax.fill_between(human_angular_vel_linspace_negative, human_maximum_linear_vel_positive, y2=human_q1_vel_linspace,
#                     lw=0,
#                 alpha=alpha_red,
#                 color='C3',
#                 hatch=red_patern
#                 )
# ax.fill_between(human_angular_vel_linspace_positive, human_maximum_linear_vel_positive, y2=human_q2_vel_linspace,
#                     lw=0,
#                 alpha=alpha_red,
#                 color='C3',
#                 hatch=red_patern
#                 )

# ::: characterized input space ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
char_angular_vel_linspace_negative = np.linspace(maximum_angular_vel_negative, 0, int(num_points / 2)).flatten()
char_angular_vel_linspace_positive = np.linspace(0, maximum_angular_vel_positive, int(num_points / 2)).flatten()
char_q1_vel_linspace = np.linspace(0, maximum_linear_vel_positive, int(num_points / 2)).flatten()
char_q2_vel_linspace = np.linspace(maximum_linear_vel_positive, 0, int(num_points / 2)).flatten()
char_q3_vel_linspace = np.linspace(maximum_linear_vel_negative, 0, int(num_points / 2)).flatten()
char_q4_vel_linspace = np.linspace(0, maximum_linear_vel_negative, int(num_points / 2)).flatten()

# ... plot characterized input space .......................................................................
q1_char_input_space = ax.plot(char_angular_vel_linspace_negative, char_q1_vel_linspace, color='C1', lw=line_width,
                              label='Characterized')
q2_char_input_space = ax.plot(char_angular_vel_linspace_positive, char_q2_vel_linspace, color='C1', lw=line_width)
q3_char_input_space = ax.plot(char_angular_vel_linspace_positive, char_q3_vel_linspace, color='C1', lw=line_width)
q4_char_input_space = ax.plot(char_angular_vel_linspace_negative, char_q4_vel_linspace, color='C1', lw=line_width)
ax.fill_between(char_angular_vel_linspace_negative, char_q1_vel_linspace, y2=char_q4_vel_linspace, alpha=alpha,
                lw=0,
                color='C1')
ax.fill_between(char_angular_vel_linspace_positive, char_q2_vel_linspace, y2=char_q3_vel_linspace, alpha=alpha,
                lw=0,
                color='C1')

# === characterized gravel =================================================================================
gravel_angular_vel_linspace_negative = np.linspace(gravel_maximum_angular_vel_negative, 0,
                                                   int(num_points / 2)).flatten()
gravel_angular_vel_linspace_positive = np.linspace(0, gravel_maximum_angular_vel_positive,
                                                   int(num_points / 2)).flatten()
gravel_q1_vel_linspace = np.linspace(0, gravel_maximum_linear_vel_positive, int(num_points / 2)).flatten()
gravel_q2_vel_linspace = np.linspace(gravel_maximum_linear_vel_positive, 0, int(num_points / 2)).flatten()
gravel_q3_vel_linspace = np.linspace(gravel_maximum_linear_vel_negative, 0, int(num_points / 2)).flatten()
gravel_q4_vel_linspace = np.linspace(0, gravel_maximum_linear_vel_negative, int(num_points / 2)).flatten()

# ... plot characterized gravel ............................................................................
q1_gravel_input_space = ax.plot(gravel_angular_vel_linspace_negative, gravel_q1_vel_linspace, color='green',
                                lw=line_width, label='Characterized')
q2_gravel_input_space = ax.plot(gravel_angular_vel_linspace_positive, gravel_q2_vel_linspace, color='green',
                                lw=line_width)
q3_gravel_input_space = ax.plot(gravel_angular_vel_linspace_positive, gravel_q3_vel_linspace, color='green',
                                lw=line_width)
q4_gravel_input_space = ax.plot(gravel_angular_vel_linspace_negative, gravel_q4_vel_linspace, color='green',
                                lw=line_width)
ax.fill_between(gravel_angular_vel_linspace_negative, gravel_q1_vel_linspace, y2=gravel_q4_vel_linspace, alpha=alpha,
                lw=0,
                color='green')
ax.fill_between(gravel_angular_vel_linspace_positive, gravel_q2_vel_linspace, y2=gravel_q3_vel_linspace, alpha=alpha,
                lw=0,
                color='green')

# === characterized snow ===================================================================================
snow_angular_vel_linspace_negative = np.linspace(snow_maximum_angular_vel_negative, 0, int(num_points / 2)).flatten()
snow_angular_vel_linspace_positive = np.linspace(0, snow_maximum_angular_vel_positive, int(num_points / 2)).flatten()
snow_q1_vel_linspace = np.linspace(0, snow_maximum_linear_vel_positive, int(num_points / 2)).flatten()
snow_q2_vel_linspace = np.linspace(snow_maximum_linear_vel_positive, 0, int(num_points / 2)).flatten()
snow_q3_vel_linspace = np.linspace(snow_maximum_linear_vel_negative, 0, int(num_points / 2)).flatten()
snow_q4_vel_linspace = np.linspace(0, snow_maximum_linear_vel_negative, int(num_points / 2)).flatten()

# ... plot characterized snow ..............................................................................
q1_snow_input_space = ax.plot(snow_angular_vel_linspace_negative, snow_q1_vel_linspace,
                              alpha=alpha_snow,
                              color=color_snow, lw=line_width,
                              label='Characterized')
q2_snow_input_space = ax.plot(snow_angular_vel_linspace_positive, snow_q2_vel_linspace,
                              alpha=alpha_snow,
                              color=color_snow, lw=line_width)
q3_snow_input_space = ax.plot(snow_angular_vel_linspace_positive, snow_q3_vel_linspace,
                              alpha=alpha_snow,
                              color=color_snow, lw=line_width)
q4_snow_input_space = ax.plot(snow_angular_vel_linspace_negative, snow_q4_vel_linspace,
                              alpha=alpha_snow,
                              color=color_snow, lw=line_width)
ax.fill_between(snow_angular_vel_linspace_negative, snow_q1_vel_linspace, y2=snow_q4_vel_linspace,
                lw=0,
                alpha=alpha_snow,
                color=color_snow)
ax.fill_between(snow_angular_vel_linspace_positive, snow_q2_vel_linspace, y2=snow_q3_vel_linspace,
                lw=0,
                alpha=alpha_snow,
                color=color_snow)

# === icp velocities =======================================================================================
for i in range(steady_state_idd_body_vel_yaw.shape[0]):
    if abs(np.median(steady_state_icp_body_vel_x[i]) - steady_state_idd_body_vel_x[i][0]) > 2.5:
        continue
    ax.scatter(steady_state_encoder_body_vel_yaw[i], steady_state_encoder_body_vel_x[i],
               s=dots_size,
               color="C1",
               edgecolors=dots_edgecolors,
               alpha=alpha_dots
               )
    ax.scatter(steady_state_icp_body_vel_yaw[i], steady_state_icp_body_vel_x[i],
               s=dots_size,
               color="green",
               edgecolors=dots_edgecolors,
               alpha=alpha_dots
               )
    # ax.quiver(steady_state_idd_body_vel_yaw[i][0], steady_state_idd_body_vel_x[i][0], np.median(
    # steady_state_icp_body_vel_yaw[i]) - steady_state_idd_body_vel_yaw[i][0], np.median(steady_state_icp_body_vel_x[
    # i]) - steady_state_idd_body_vel_x[i][0], angles='xy', scale_units='xy', scale=1, width=0.03, headwidth=5,
    # headlength=3, headaxislength=3, minlength=1, minshaft=1, units='xy', color="C3", alpha=0.5)
    # for j in range(steady_state_idd_body_vel_yaw.shape[1]):
    # ax.quiver(steady_state_idd_body_vel_yaw[i][j], steady_state_idd_body_vel_x[i][j],
    # steady_state_icp_body_vel_yaw[i][j] - steady_state_idd_body_vel_yaw[i][j], steady_state_icp_body_vel_x[i][j] -
    # steady_state_idd_body_vel_x[i][j], angles='xy', scale_units='xy', scale=1, width=0.03, headwidth=3,
    # headlength=1, headaxislength=1, minlength=1, minshaft=1, units='xy', alpha=0.1)

for i in range(steady_state_icp_body_vel_x_tracks.shape[0]):
    ax.scatter(steady_state_icp_body_vel_yaw_tracks[i], steady_state_icp_body_vel_x_tracks[i],
               s=dots_size,
               color=color_snow,
               edgecolors=dots_edgecolors,
               alpha=alpha_dots_snow
               )

# # === humand-commanded velocities ==========================================================================
# ax.scatter(cmd_vel_omega, cmd_vel_x, s=dots_size, color="gray",
#       edgecolors = dots_edgecolors,
#       alpha=alpha_dots,
#       )

# ::: Add dougnut calib aerial imagerie ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# from PIL import Image
# img_aerial_gravel = np.asarray(Image.open("./aerial_photo/dougnut_calib_warthog_gravel.png"))
#
# ax_li = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE', zorder=-1)
# ax_li.imshow(img_aerial_gravel)
# ax_li.axis('off')


# === Show =================================================================================================

# # ... Legend version 2 .....................................................................................
# legend_elements = [
#         Rectangle((0, 0), width=5, height=3,
#                   label='Naive', linestyle='solid',
#                   ##...Full version...
#                   alpha=0.8,
#                   edgecolor='k',
#                   facecolor="lightgray",
#                   ##...Pattern version...
#                   # alpha=0.4,
#                   # edgecolor='grey',
#                   # facecolor="none",
#                   # hatch=naive_patern,
#                   ),
#         Rectangle((0, 0), width=5, height=3,
#                   label='Human', linestyle='solid',
#                   ##...Full version...
#                   alpha=0.8,
#                   edgecolor='k',
#                   facecolor="C3",
#                   ##...Pattern version...
#                   # alpha=0.7,
#                   # edgecolor='C3',
#                   # facecolor="none",
#                   # hatch=red_patern,
#                   ),
#         Rectangle((0, 0), width=5, height=3, facecolor="C1", label='Powertrain', linestyle='solid',
#                   edgecolor='k'
#                   # edgecolor='C1'
#                   ),
#         # Rectangle((0,0), width=5, height=3, facecolor="gray", label='Human', linestyle='solid',
#         #         # edgecolor='k'
#         # edgecolor='gray'
#         #                 ),
#         #  Rectangle((0,0), width=5, height=3, facecolor="C3", label='Doughnut', linestyle='solid',
#         #         #  edgecolor='k'
#         #  edgecolor='C3'
#         #         ),
#         Rectangle((0, 0), width=5, height=3, facecolor="green", label='Gravel', linestyle='solid',
#                   edgecolor='k'
#                   # edgecolor='green'
#                   ),
#         Rectangle((0, 0), width=5, height=3, facecolor="C0", label='Snow', linestyle='solid',
#                   edgecolor='k'
#                   # edgecolor='C0'
#                   ), ]
#
# fig.legend(handles=legend_elements, loc="lower center",
#            # bbox_to_anchor=(0.5, -0.019),
#            bbox_to_anchor=(0.5, 0.03),
#            ncol=5,
#            columnspacing=1.0,
#            handletextpad=0.6,
#            handlelength=1.25,
#            prop={ 'size': 'small' })

# # ... Legend version 3 .....................................................................................
# legend_elements = [
#         Rectangle((0, 0), width=5, height=3, linestyle='solid', edgecolor='k', facecolor="lightgray", alpha=0.8,
#                   label='Naive',
#                   ),
#         Rectangle((0, 0), width=5, height=3, facecolor="C1", linestyle='solid', edgecolor='k',
#                   label='Encoders',
#                   ),
#         Rectangle((0, 0), width=5, height=3, facecolor="green", linestyle='solid', edgecolor='k',
#                   label='Gravel',
#                   ),
#         Rectangle((0, 0), width=5, height=3, facecolor="C0", linestyle='solid', edgecolor='k',
#                   label='Snow',
#                   ),
#         Rectangle((0, 0), width=5, height=3, linestyle='solid', edgecolor='k', facecolor="C3", alpha=0.8,
#                   label='Typical human driving',
#                   ),
#         ]
#
# legend_order = [
#         0,
#         3,
#         1,
#         2,
#         4,
#         ]
#
# fig.legend(handles=[legend_elements[idx] for idx in legend_order], loc="lower center",
#            bbox_to_anchor=(0.5, 0.07),
#            borderpad=0.5,
#            ncol=4,
#            columnspacing=0.5,
#            handletextpad=0.6,
#            handlelength=1.25,
#            prop={ 'size': 'small' })

# ax.set_xlabel("Commanded angular velocity [rad/s]")
# ax.set_ylabel("Commanded linear velocity [m/s]")


fig.savefig("fig_2_upgrade.pdf", transparent=True)  # Does not render face pattern
fig.savefig("fig_2_upgrade.jpg", dpi='figure')
fig.savefig("fig_2_upgrade_600dpi.png", dpi=600, transparent=True)
plt.show()
