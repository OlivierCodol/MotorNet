"""This module contains various functions for plotting data from `MotorNet` training and simulation sessions.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import animation
from matplotlib.collections import LineCollection


def compute_limits(data, margin=0.1):
    """This small function computes the limits to use for plotting data, given the range of the dataset and a margin
    size around that range.

    Args:
        data: A `numpy.ndarray` object containing the data to plot.
        margin: Float, the proportion of the data's range to add as margin for plotting. For instance, if the data
            value range from `0` to `10`, and the margin is set to `0.2`, then the limits would become `[-2, 12]` since
            the range is `10`.

    Returns:
        A list of two `float` values, representing the lower and upper limits to use on the plot, in that order.
    """
    data_range = data.ptp()
    m = data_range * margin
    minval = np.min(data) - m
    maxval = np.max(data) + m
    return minval, maxval


def _plot_line_collection(segments, cmap: str = 'viridis', linewidth: int = 1, figure=plt.gcf(), **kwargs):
    n_gradient = kwargs.get('n_gradient', segments.shape[0])

    norm = plt.Normalize(0, n_gradient)  # Create a continuous norm to map from data points to colors
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(0, n_gradient))  # Set the values used for colormapping
    lc.set_linewidth(linewidth)

    axes = plt.gca()
    axes.add_collection(lc)

    clb = figure.colorbar(lc, ax=axes)
    clb.set_label('timestep')
    return axes


def plot_pos_over_time(cart_results):
    """Plot position results over time, giving a darker color to the early part of a trajectory, and a lighter color to
    the later part of the trajectory.

    Args:
        cart_results: A `numpy.ndarray` object containing the trajectories. Its dimensionality should be
            `n_batches * n_timesteps * (2 . n_dim)`, with `n_dim` being the trajectory's dimensionality. For instance,
            for a planar reach, since the movement is in 2D space, we have`n_dim = 2`, meaning the third dimension of
            the array is `4` (x position, y position, x velocity, y velocity).
    """
    n_timesteps = cart_results.shape[1]
    segments, points = _results_to_line_collection(cart_results)
    axes = _plot_line_collection(segments, n_gradient=n_timesteps - 1)
    axes.set_xlabel('cartesian x')
    axes.set_ylabel('cartesian y')
    axes.set_aspect('equal', adjustable='box')
    # plt.scatter(0., 0., label='shoulder fixation', zorder=np.inf, marker='+')


def _results_to_line_collection(results):
    # each line is a segment of the trajectory (a sample), and it will have its own colour from the gradent
    # each line has two values (per dimension): start point and end point
    # n_samples * 1 * space_dim * batch_size
    results_pos, _ = np.split(results, 2, axis=-1)
    space_dim = results_pos.shape[-1]
    points = results_pos[:, :, :, np.newaxis].swapaxes(0, -1).swapaxes(0, 1)
    # (n_samples-1) * 2 * space_dim * batch_size
    segments_by_batch = np.concatenate([points[:-1], points[1:]], axis=1)
    # concatenate batch and time dimensions (b1t1, b2t1, ..., b1t2, b2t2, ....,b1tn, b2tn, ..., bntn)
    # n_lines * 2 * space_dim
    segments_all_batches = np.moveaxis(segments_by_batch, -1, 0).reshape((-1, 2, space_dim))
    return segments_all_batches, points


def plot_arm26_over_time(arm, joint_state, cmap: str = 'viridis', linewidth: int = 1, figure=plt.gcf(), **kwargs):
    """Plot an arm26 over time, with earlier and later arm configuration in the movement being represented as darker and
    brighter colors, respectively.

    Args:
        arm: `motornet.plants.skeletons.Arm26` object to plot.
        joint_state: A `numpy.ndarray` object containing the trajectory. Its dimensionality should be
            `1 * n_timesteps * (2 . n_dim)`, with `n_dim` being the trajectory's dimensionality. For an `arm26`,
            since the arm has 2 degrees of freedom, we have`n_dim = 2`, meaning the third dimension of
            the array is `4` (shoulder position, elbow position, shoulder velocity, elbow velocity).
        cmap: String, colormap supported by `matplotlib`.
        figure: `matplotlib.figure.Figure` handle.
        linewidth: Integer, line width of the arm segments being plotted.
    """

    assert joint_state.shape[0] == 1  # can only take one simulation at a time
    n_timesteps = joint_state.shape[1]
    joint_pos = np.moveaxis(joint_state, 0, -1).squeeze()

    joint_angle_sum = joint_pos[:, 0] + joint_pos[:, 1]
    elb_pos_x = arm.L1 * np.cos(joint_pos[:, 0])
    elb_pos_y = arm.L1 * np.sin(joint_pos[:, 0])
    end_pos_x = elb_pos_x + arm.L2 * np.cos(joint_angle_sum)
    end_pos_y = elb_pos_y + arm.L2 * np.sin(joint_angle_sum)

    upper_arm_x = np.stack([np.zeros_like(elb_pos_x), elb_pos_x], axis=1)
    upper_arm_y = np.stack([np.zeros_like(elb_pos_y), elb_pos_y], axis=1)
    upper_arm = np.stack([upper_arm_x, upper_arm_y], axis=2)

    lower_arm_x = np.stack([elb_pos_x, end_pos_x], axis=1)
    lower_arm_y = np.stack([elb_pos_y, end_pos_y], axis=1)
    lower_arm = np.stack([lower_arm_x, lower_arm_y], axis=2)

    segments = np.squeeze(np.concatenate([upper_arm, lower_arm], axis=0))
    _, axes, clb = _plot_line_collection(
        segments, cmap=cmap, linewidth=linewidth, figure=figure,
        n_gradient=n_timesteps, **kwargs)

    axes.set_xlim(compute_limits(segments[:, :, 0]))
    axes.set_ylim(compute_limits(segments[:, :, 1]))
    axes.set_xlabel('cartesian x')
    axes.set_ylabel('cartesian y')
    axes.set_aspect('equal', adjustable='box')


def plot_opensim(results, save_path):
    j = results['joint position'].numpy().transpose(2, 1, 0).reshape((4, -1), order='F').transpose()
    m = results['muscle state'].numpy().transpose(2, 3, 1, 0).reshape((5, 6, -1), order='F').transpose(2, 1, 0)
    mdict = {'muscle': m[:, :, 0], 'joint': j[:, 0:2]}
    scipy.io.savemat(save_path + '.mat', mdict)

    variable = 'asd'
    with open(save_path + '.sto', "w") as text_file:
        print(f"Stuff: {variable}", file=text_file)


def animate_arm_trajectory(joint_position, plant, path_name='./Arm_animation.mp4'):
    assert joint_position.shape[0] == 1
    joint_position = tf.reshape(joint_position, (-1, plant.state_dim))

    plant = plant.skeleton

    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
    line, = ax.plot([], [], lw=2, alpha=0.7, color='red')  # Movement path
    L1, = ax.plot([], [], lw=4)  # L1
    L2, = ax.plot([], [], lw=4)  # L1

    # plt.title('Desired Title')
    # Axis control 
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')  # Make axis equal
    plt.style.use('dark_background')

    def my_joint2cartesian(plant, joint_pos):
        joint_pos = tf.reshape(joint_pos, (-1, plant.state_dim))
        joint_angle_sum = joint_pos[:, 0] + joint_pos[:, 1]

        c1 = tf.cos(joint_pos[:, 0])
        s1 = tf.sin(joint_pos[:, 0])
        c12 = tf.cos(joint_angle_sum)
        s12 = tf.sin(joint_angle_sum)

        end_pos_x_l1 = plant.L1 * c1
        end_pos_y_l1 = plant.L1 * s1
        end_pos_x_l2 = plant.L2 * c12
        end_pos_y_l2 = plant.L2 * s12

        return end_pos_x_l1, end_pos_y_l1, end_pos_x_l2, end_pos_y_l2

    # Initialization of data lists
    def init():
        # creating void frame 
        line.set_data([], [])
        L1.set_data([], [])
        L2.set_data([], [])
        ax.scatter([0], [0])
        return line,

        # Empty List for trajectories and arm position

    xdata, ydata = [], []
    L1_xdata, L1_ydata = [], []
    L2_xdata, L2_ydata = [], []

    # animation
    def animate(i):
        # Get the position of end_point, L1, and L2
        end_pos_x_l1, end_pos_y_l1, end_pos_x_l2, end_pos_y_l2 = my_joint2cartesian(plant, joint_position[i:i + 1, :])

        # Append the endpoint position 
        xdata.append(end_pos_x_l1 + end_pos_x_l2)
        ydata.append(end_pos_y_l1 + end_pos_y_l2)
        line.set_data(xdata, ydata)

        # Append the L1 position 
        L1_xdata = [0, end_pos_x_l1]
        L1_ydata = [0, end_pos_y_l1]
        L1.set_data(L1_xdata, L1_ydata)

        # Append the L2 position
        L2_xdata = [end_pos_x_l1, end_pos_x_l1 + end_pos_x_l2]
        L2_ydata = [end_pos_y_l1, end_pos_y_l1 + end_pos_y_l2]
        L2.set_data(L2_xdata, L2_ydata)

        return line, L1, L2

    # call animation	 
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=joint_position.shape[0], interval=plant.dt, blit=True)

    # save the animated file, (Used pillow, since it is usually installed by default)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1. / plant.dt)
    anim.save(path_name, writer=writer, dpi=100)
