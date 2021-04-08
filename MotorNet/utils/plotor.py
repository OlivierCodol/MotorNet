import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def compute_limits(data):
    data_range = data.ptp()
    margin = data_range * 0.1
    minval = np.min(data) - margin
    maxval = np.max(data) + margin
    return minval, maxval


def plot_line_collection(segments, **kwargs):
    n_gradient = kwargs.get('n_gradient', segments.shape[0])
    cmap = kwargs.get('cmap', 'viridis')
    figure = kwargs.get('figure', plt.gcf())
    linewidth = kwargs.get('linewidth', 1)

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
    n_timesteps = cart_results.shape[1]
    segments, points = results_to_line_collection(cart_results)
    axes = plot_line_collection(segments, n_gradient=n_timesteps - 1)
    axes.set_xlabel('cartesian x')
    axes.set_ylabel('cartesian y')
    axes.set_aspect('equal', adjustable='box')
    plt.scatter(0., 0., label='shoulder fixation', zorder=np.inf, marker='+')


def results_to_line_collection(results):
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


def plot_arm_over_time(arm, joint_results, **kwargs):
    assert joint_results.shape[0] == 1  # can only take one simulation at a time
    n_timesteps = joint_results.shape[1]
    joint_pos = np.moveaxis(joint_results, 0, -1).squeeze()

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
    _, axes, clb = plot_line_collection(segments, n_gradient=n_timesteps, **kwargs)
    axes.set_xlim(compute_limits(segments[:, :, 0]))
    axes.set_ylim(compute_limits(segments[:, :, 1]))
    axes.set_xlabel('cartesian x')
    axes.set_ylabel('cartesian y')
    axes.set_aspect('equal', adjustable='box')
