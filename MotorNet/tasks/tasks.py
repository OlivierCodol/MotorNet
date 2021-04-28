import numpy as np
from abc import ABC, abstractmethod
from MotorNet.nets.losses import position_loss, activation_squared_loss, empty_loss
import tensorflow as tf


class Task(ABC):
    def __init__(self, plant, n_timesteps=1, batch_size=1, task_args=None):
        if task_args is None:
            task_args = {}
        self.plant = plant
        self.initial_joint_state = None
        self.task_args = task_args
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.losses = {}
        self.loss_weights = {}

    @abstractmethod
    def generate(self):
        return

    def get_input_dim(self):
        [inputs, _] = self.generate()
        shape = inputs.get_shape().as_list()
        return shape[-1]

    def get_losses(self):
        return [self.losses, self.loss_weights]

    def get_perturbation_dim_start(self):
        return None

    def get_initial_state(self):
        if self.initial_joint_state is not None:
            return np.expand_dims(self.initial_joint_state, axis=0)
        else:
            return None


class TaskStaticTarget(Task):
    def __init__(self, plant, n_timesteps=5000, batch_size=1, task_args=None):
        super().__init__(plant, n_timesteps, batch_size, task_args)
        # define losses and loss weights for this task
        self.losses = {'cartesian position': position_loss(), 'muscle state': activation_squared_loss()}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.01}
        self.initial_joint_state = np.deg2rad([45., 90., 0., 0.])

    def generate(self, **kwargs):
        self.n_timesteps = kwargs.get('n_timesteps', self.n_timesteps)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        goal_states = self.plant.draw_random_uniform_states(batch_size=self.batch_size)
        targets = self.plant.state2target(state=self.plant.joint2cartesian(goal_states), n_timesteps=self.n_timesteps)
        return [targets[:, :, 0:2], targets]

class TaskStaticTargetWithPerturbations(Task):
    def __init__(self, plant, n_timesteps=5000, batch_size=1, task_args=None):
        super().__init__(plant, n_timesteps, batch_size, task_args)
        # define losses and loss weights for this task
        self.losses = {'cartesian position': position_loss(), 'muscle state': activation_squared_loss()}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.01}
        self.initial_joint_state = np.deg2rad([45., 90., 0., 0.])

    def generate(self, **kwargs):
        self.n_timesteps = kwargs.get('n_timesteps', self.n_timesteps)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        goal_states = self.plant.draw_random_uniform_states(batch_size=self.batch_size)
        targets = self.plant.state2target(state=self.plant.joint2cartesian(goal_states), n_timesteps=self.n_timesteps)
        perturbations = tf.constant(3., shape=(self.batch_size, self.n_timesteps, 2))
        return [tf.concat([targets[:, :, 0:2], perturbations], axis=2), targets]

    def get_perturbation_dim_start(self):
        return 2


class TaskDelayedReach(Task):
    def __init__(self, plant, n_timesteps=5000, batch_size=1, task_args=None):
        super().__init__(plant, n_timesteps, batch_size, task_args)
        # define losses and loss weights for this task
        self.losses = {'cartesian position': position_loss(), 'muscle state': activation_squared_loss()}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.01}
        self.initial_joint_state = np.deg2rad([45., 90., 0., 0.])

        if "bump_length" in self.task_args:
            self.bump_length = int(self.task_args['bump_length'] / 1000 / plant.dt)
        else:
            self.bump_length = int(50 / 1000 / plant.dt)
        if "bump_height" in self.task_args:
            self.bump_height = self.task_args['bump_height']
        else:
            self.bump_height = 1
        if "delay_range" in self.task_args:
            self.delay_range = np.array(self.task_args['delay_range']) / 1000 / plant.dt
        else:
            self.delay_range = np.array([100, 900]) / 1000 / plant.dt

    def generate(self, **kwargs):
        self.n_timesteps = kwargs.get('n_timesteps', self.n_timesteps)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        goal_states = self.plant.draw_random_uniform_states(batch_size=self.batch_size)
        targets = self.plant.state2target(state=self.plant.joint2cartesian(goal_states),
                                          n_timesteps=self.n_timesteps).numpy()
        temp_center = self.plant.get_initial_state(batch_size=1, joint_state=self.get_initial_state())
        center = temp_center[1][0, :]
        temp_inputs = []
        for i in range(self.batch_size):
            delay_time = generate_delay_time(self.delay_range[0], self.delay_range[1], 'random')
            bump = np.concatenate([np.zeros(delay_time), np.ones(self.bump_length)*self.bump_height,
                                   np.zeros(int(self.n_timesteps - delay_time - self.bump_length))])
            temp_inputs.append(np.concatenate([np.squeeze(targets[i, :, 0:2]), np.expand_dims(bump, axis=1)], axis=1))
            targets[i, 0:delay_time, :] = center

        inputs = tf.stack(temp_inputs, axis=0)
        return [inputs, tf.convert_to_tensor(targets)]


def generate_delay_time(delay_min, delay_max, delay_mode):
    if delay_mode == 'random':
        delay_time = np.random.uniform(delay_min, delay_max)
    elif delay_mode == 'noDelayInput':
        delay_time = 0
    else:
        raise AttributeError

    return int(delay_time)
