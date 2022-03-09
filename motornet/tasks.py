import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from abc import abstractmethod
from motornet.nets.losses import PositionLoss, L2ActivationLoss, L2xDxActivationLoss, L2xDxRegularizer


class Task(tf.keras.utils.Sequence):
    """
    Base class for tasks.
    """
    def __init__(self, network, initial_joint_state=None, **kwargs):
        self.__name__ = 'Generic Task'
        self.network = network
        self.training_iterations = 1000
        self.training_batch_size = 32
        self.training_n_timesteps = 100
        self.delay_range = [0, 0]
        self.do_recompute_targets = False
        self.kwargs = kwargs
        self.losses = {name: None for name in self.network.output_names}
        self.loss_names = {name: name for name in self.network.output_names}
        self.loss_weights = {name: 0. for name in self.network.output_names}

        if initial_joint_state is not None:
            initial_joint_state = np.array(initial_joint_state)
            if len(initial_joint_state.shape) == 1:
                initial_joint_state = initial_joint_state.reshape(1, -1)
            self.n_initial_joint_states = initial_joint_state.shape[0]
            self.initial_joint_state_original = initial_joint_state.tolist()
        else:
            self.initial_joint_state_original = None
            self.n_initial_joint_states = None
        self.initial_joint_state = initial_joint_state

        self.convert_to_tensor = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(x))

    def add_loss(self, assigned_output, loss, loss_weight=1.):
        self.losses[assigned_output] = loss
        self.loss_weights[assigned_output] = loss_weight
        if hasattr(loss, 'name'):
            self.loss_names[assigned_output] = loss.name

    @abstractmethod
    def generate(self, batch_size, n_timesteps):
        return

    def get_initial_state(self, batch_size, joint_state=None):
        if joint_state is None:
            if self.initial_joint_state is None:
                inputs = None
            else:
                i = np.random.randint(0, self.n_initial_joint_states, batch_size)
                inputs = self.initial_joint_state[i, :]
            initial_states = self.network.get_initial_state(batch_size=batch_size, inputs=inputs)
        else:
            initial_states = self.network.get_initial_state(batch_size=batch_size, inputs=joint_state)
        return initial_states

    def get_input_dim(self):
        [inputs, _, _] = self.generate(batch_size=1, n_timesteps=self.delay_range[-1]+1)

        def sort_shape(i):
            if tf.is_tensor(i):
                s = i.get_shape().as_list()
            else:
                s = i.shape
            return s[-1]

        if type(inputs) is dict:
            shape = {key: sort_shape(val) for key, val in inputs.items()}
        else:
            shape = inputs

        return shape

    def get_losses(self):
        return [self.losses, self.loss_weights]

    def set_training_params(self, batch_size, n_timesteps, iterations):
        self.training_batch_size = batch_size
        self.training_n_timesteps = n_timesteps
        self.training_iterations = iterations

    def get_save_config(self):
        cfg = {'task_kwargs': self.kwargs, 'name': self.__name__, 'training_iterations': self.training_iterations,
               'training_batch_size': self.training_batch_size, 'training_n_timesteps': self.training_n_timesteps,
               'do_recompute_targets': self.do_recompute_targets, 'loss_weights': self.loss_weights,
               'initial_joint_state': self.initial_joint_state_original}
        return cfg

    def __getitem__(self, idx):
        [inputs, targets, init_states] = self.generate(batch_size=self.training_batch_size,
                                                       n_timesteps=self.training_n_timesteps)
        return [inputs, init_states], targets

    def __len__(self):
        return self.training_iterations

    def get_input_dict_layers(self):
        return {key: Input((None, val,), name=key) for key, val in self.get_input_dim().items()}

    def get_initial_state_layers(self):
        n_muscles = self.network.plant.n_muscles
        state0 = [
            Input((self.network.plant.output_dim,), name='joint0'),
            Input((self.network.plant.output_dim,), name='cartesian0'),
            Input((self.network.plant.muscle_state_dim, n_muscles,), name='muscle0'),
            Input((self.network.plant.geometry_state_dim, n_muscles,), name='geometry0'),
            Input((n_muscles * 2, self.network.plant.proprioceptive_delay,), name='proprio_feedback0'),
            Input((self.network.plant.space_dim, self.network.plant.visual_delay,), name='visual_feedback0')
        ]
        state0.extend([Input((n,), name='gru' + str(k) + '_hidden0') for k, n in enumerate(self.network.n_units)])
        return state0


class RandomTargetReach(Task):
    def __init__(self, network, **kwargs):
        super().__init__(network, **kwargs)
        self.__name__ = 'RandomTargetReach'
        max_iso_force = self.network.plant.muscle.max_iso_force
        self.add_loss('muscle state', loss=L2ActivationLoss(max_iso_force=max_iso_force), loss_weight=.2)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)

    def generate(self, batch_size, n_timesteps):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
        goal_states = self.network.plant.joint2cartesian(goal_states_j)
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs = {"inputs": targets[:, :, :self.network.plant.space_dim]}
        return [inputs, targets, init_states]


class RandomTargetReachWithLoads(Task):
    def __init__(self, network, endpoint_load: float, **kwargs):
        super().__init__(network, **kwargs)
        self.__name__ = 'RandomTargetReachWithLoads'
        max_iso_force = self.network.plant.muscle.max_iso_force
        self.add_loss('muscle state', loss=L2ActivationLoss(max_iso_force=max_iso_force), loss_weight=.2)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)
        self.endpoint_load = endpoint_load

    def generate(self, batch_size, n_timesteps):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
        goal_states = self.network.plant.joint2cartesian(goal_states_j)
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        endpoint_load = tf.constant(self.endpoint_load, shape=(batch_size, n_timesteps, 2))
        inputs = {"inputs": targets[:, :, :self.network.plant.space_dim], "endpoint_load": endpoint_load}
        return [inputs, targets, init_states]


class DelayedReach(Task):
    """
    A random-delay reach to a random target from a random starting position.

    Args:
      delay_range: Two-items list or numpy.array that indicate the minimum and maximum value of the delay timer.
        The delay is randomly drawn from a uniform distribution bounded by these values.
    """
    def __init__(self, network, **kwargs):
        super().__init__(network, **kwargs)
        self.__name__ = 'DelayedReach'
        max_iso_force = self.network.plant.muscle.max_iso_force
        self.add_loss('muscle state', loss=L2ActivationLoss(max_iso_force=max_iso_force), loss_weight=.2)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)
        delay_range = np.array(kwargs.get('delay_range', [0.3, 0.6])) / self.network.plant.dt
        self.delay_range = [int(delay_range[0]), int(delay_range[1])]
        self.convert_to_tensor = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(x))

    def generate(self, batch_size, n_timesteps):
        goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
        goal_states = self.network.plant.joint2cartesian(goal_states_j)
        init_states = self.get_initial_state(batch_size=batch_size)
        center = self.network.plant.joint2cartesian(init_states[0][:, :])
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()

        inputs = copy.deepcopy(targets)
        gocue = np.zeros([batch_size, n_timesteps, 1])
        for i in range(batch_size):
            delay_time = int(np.random.uniform(self.delay_range[0], self.delay_range[1]))
            targets[i, :delay_time, :] = center[i, np.newaxis, :]
            gocue[i, delay_time, 0] = 1.

        inputs = {"inputs": np.concatenate([inputs, gocue], axis=-1)}
        return [inputs, self.convert_to_tensor(targets), init_states]


class CentreOutReach(Task):
    def __init__(self, network, **kwargs):
        super().__init__(network, **kwargs)
        self.__name__ = 'CentreOutReach'

        self.angle_step = kwargs.get('reach_angle_step_deg', 15)
        self.catch_trial_perc = kwargs.get('catch_trial_perc', 33)
        self.reaching_distance = kwargs.get('reaching_distance', 0.1)
        self.start_position = kwargs.get('start_joint_position', None)
        if not self.start_position:
            # start at the center of the workspace
            lb = np.array(self.network.plant.pos_lower_bound)
            ub = np.array(self.network.plant.pos_upper_bound)
            self.start_position = lb + (ub - lb) / 2
        self.start_position = np.array(self.start_position).reshape(1, -1)

        deriv_weight = kwargs.get('deriv_weight', 0.05)
        max_iso_force = self.network.plant.muscle.max_iso_force
        dt = self.network.plant.dt
        muscle_loss = L2xDxActivationLoss(max_iso_force=max_iso_force, dt=dt, deriv_weight=deriv_weight)
        gru_loss = L2xDxRegularizer(deriv_weight=.01, dt=self.network.plant.dt)
        self.add_loss('gru_hidden0', loss_weight=0.75, loss=gru_loss)
        self.add_loss('muscle state', loss_weight=.2, loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=1., loss=PositionLoss())

        go_cue_range = np.array(kwargs.get('go_cue_range', [0.05, 0.25])) / dt
        self.go_cue_range = [int(go_cue_range[0]), int(go_cue_range[1])]
        self.delay_range = self.go_cue_range

    def generate(self, batch_size, n_timesteps, **kwargs):
        catch_trial = np.zeros(batch_size, dtype='float32')
        validation = kwargs.get('validation', False)
        if not validation:
            init_states = self.get_initial_state(batch_size=batch_size)
            goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
            goal_states = self.network.plant.joint2cartesian(goal_states_j)
            p = int(np.floor(batch_size * self.catch_trial_perc / 100))
            catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
        else:
            angle_set = np.deg2rad(np.arange(0, 360, self.angle_step))
            reps = int(np.ceil(batch_size / len(angle_set)))
            angle = np.tile(angle_set, reps=reps)
            batch_size = reps * len(angle_set)
            catch_trial = np.zeros(batch_size, dtype='float32')

            start_jpv = np.concatenate([self.start_position, np.zeros_like(self.start_position)], axis=1)
            start_cpv = self.network.plant.joint2cartesian(start_jpv)
            end_cp = self.reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)

        center = self.network.plant.joint2cartesian(init_states[0][:, :])
        go_cue = np.zeros([batch_size, n_timesteps, 1])
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs = copy.deepcopy(targets[:, :, :self.network.plant.space_dim])

        for i in range(batch_size):
            if not validation:
                go_cue_time = int(np.random.uniform(self.go_cue_range[0], self.go_cue_range[1]))
            else:
                go_cue_time = int(self.go_cue_range[0] + np.diff(self.go_cue_range) / 2)

            if catch_trial[i] > 0.:
                targets[i, :, :] = center[i, np.newaxis, :]
            else:
                targets[i, :go_cue_time, :] = center[i, np.newaxis, :]
                go_cue[i, go_cue_time:, 0] = 1.

        inputs = {"inputs": np.concatenate([inputs, go_cue], axis=-1)}
        return [inputs, self.convert_to_tensor(targets), init_states]
