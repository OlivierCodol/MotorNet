
import copy
import numpy as np
import tensorflow as tf
import random
from abc import abstractmethod
from scipy.signal import butter, lfilter
from MotorNet.nets.losses import PositionLoss, ActivationSquaredLoss, ActivationVelocitySquaredLoss, \
    ActivationDiffSquaredLoss, L2Regularizer, RecurrentActivityRegularizer
# TODO: check that "generate" methods do not feed the memory leak


class Task(tf.keras.utils.Sequence):
    """
    Base class for tasks.
    """
    def __init__(self, controller, initial_joint_state=None, **kwargs):
        self.__name__ = 'Generic Task'
        self.controller = controller
        self.plant = controller.plant
        self.training_iterations = 1000
        self.training_batch_size = 32
        self.training_n_timesteps = 100
        self.delay_range = [0, 0]
        self.do_recompute_targets = False
        self.kwargs = kwargs
        self.losses = {name: None for name in self.controller.output_names}
        self.loss_names = {name: name for name in self.controller.output_names}
        self.loss_weights = {name: 0. for name in self.controller.output_names}

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

    def add_loss(self, assigned_output, loss, loss_weight=1.):
        self.losses[assigned_output] = loss
        self.loss_weights[assigned_output] = loss_weight
        if hasattr(loss, 'name'):
            self.loss_names[assigned_output] = loss.name

    @abstractmethod
    def generate(self, batch_size, n_timesteps, **kwargs):
        return

    def get_initial_state(self, batch_size):
        if self.initial_joint_state is None:
            inputs = None
        else:
            i = np.random.randint(0, self.n_initial_joint_states, batch_size)
            inputs = self.initial_joint_state[i, :]
        return self.controller.get_initial_state(batch_size=batch_size, inputs=inputs)

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


class TaskStaticTarget(Task):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskStaticTarget'
        max_iso_force = self.plant.Muscle.max_iso_force
        self.add_loss('muscle state', loss=ActivationSquaredLoss(max_iso_force=max_iso_force), loss_weight=.2)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states = self.plant.joint2cartesian(self.plant.draw_random_uniform_states(batch_size=batch_size))
        targets = self.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs = {"inputs": targets[:, :, :self.plant.space_dim]}
        return [inputs, targets, init_states]


class TaskStaticTargetWithPerturbations(Task):
    def __init__(self, controller, endpoint_load: float, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskStaticTargetWithPerturbations'
        max_iso_force = self.plant.Muscle.max_iso_force
        self.add_loss('muscle state', loss=ActivationSquaredLoss(max_iso_force=max_iso_force), loss_weight=.2)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)
        self.endpoint_load = endpoint_load

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states = self.plant.joint2cartesian(self.plant.draw_random_uniform_states(batch_size=batch_size))
        targets = self.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        endpoint_load = tf.constant(self.endpoint_load, shape=(batch_size, n_timesteps, 2))
        inputs = {"inputs": targets[:, :, :self.plant.space_dim], "endpoint_load": endpoint_load}
        return [inputs, targets, init_states]


class TaskDelayedReach(Task):
    """
    A random-delay reach to a random target from a random starting position.

    Args:
      delay_range: Two-items list or numpy.array that indicate the minimum and maximum value of the delay timer.
        The delay is randomly drawn from a uniform distribution bounded by these values.
    """
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskDelayedReach'
        max_iso_force = self.plant.Muscle.max_iso_force
        self.add_loss('muscle state', loss=ActivationSquaredLoss(max_iso_force=max_iso_force), loss_weight=.2)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)
        delay_range = np.array(kwargs.get('delay_range', [0.3, 0.6])) / self.plant.dt
        self.delay_range = [int(delay_range[0]), int(delay_range[1])]
        self.convert_to_tensor = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(x))

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        center = self.plant.joint2cartesian(init_states[0][:, :])
        goal_states = self.plant.joint2cartesian(self.plant.draw_random_uniform_states(batch_size=batch_size))
        targets = self.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()

        inputs = copy.deepcopy(targets)
        gocue = np.zeros([batch_size, n_timesteps, 1])
        for i in range(batch_size):
            delay_time = int(np.random.uniform(self.delay_range[0], self.delay_range[1]))
            targets[i, :delay_time, :] = center[i, np.newaxis, :]
            gocue[i, delay_time, 0] = 1.

        inputs = {"inputs": np.concatenate([inputs, gocue], axis=-1)}
        return [inputs, self.convert_to_tensor(targets), init_states]


class TaskDelayedMultiReach(Task):
    def __init__(self, controller, initial_joint_state=None, **kwargs):
        super().__init__(controller, initial_joint_state=initial_joint_state)
        self.__name__ = 'TaskDelayedMultiReach'
        max_iso_force = self.plant.Muscle.max_iso_force
        self.add_loss('muscle state', loss=ActivationSquaredLoss(max_iso_force=max_iso_force), loss_weight=.2)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)

        self.bump_length = int(kwargs.get('bump_length', 50) / 1000 / self.plant.dt)
        self.bump_height = kwargs.get('bump_height', 3)
        self.delay_range = np.array(kwargs.get('delay_range', [100, 900])) / 1000 / self.plant.dt
        self.num_target = kwargs.get('num_target', 1)

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        center = self.plant.joint2cartesian(init_states[0][0, :])

        num_target = self.num_target
        target_list = np.zeros((batch_size, n_timesteps, 4, num_target))

        for tg in range(num_target):
            goal_states = self.plant.draw_random_uniform_states(batch_size=batch_size)
            target_list[:, :, :, tg] = self.plant.state2target(
                state=self.plant.joint2cartesian(goal_states),
                n_timesteps=n_timesteps).numpy()

        temp_inputs = []
        temp_targets = []

        sequence_time = int((num_target + 1) * target_list.shape[1] + self.delay_range[1] + self.bump_length)
        targets = np.zeros((batch_size, sequence_time, 4))

        for i in range(batch_size):
            # Create Inputs
            delay_time = generate_delay_time(self.delay_range[0], self.delay_range[1], 'random')
            bump = np.concatenate([np.zeros(n_timesteps),
                                   np.zeros(delay_time),
                                   np.ones(self.bump_length) * self.bump_height,
                                   np.zeros(int(num_target * n_timesteps) + int(self.delay_range[1] - delay_time))])

            input_tensor = np.concatenate([np.squeeze(target_list[i, :, 0:2, tr]) for tr in range(num_target)], axis=1)  # Concatenate input positions for the targets
            input_tensor = tf.repeat(input_tensor, num_target, axis=0)  # Repeat the input (num_target) times
            # Concatenate zeros for delay period
            input_tensor = np.concatenate([
                np.zeros((n_timesteps, 2 * num_target)),                                  # For Go to Center
                np.zeros((int(self.delay_range[1] + self.bump_length), 2 * num_target)),  # For remain in center and see targets
                input_tensor,                                                             # For each to targets
                ], axis=0)
            input_tensor[:n_timesteps] = np.repeat(center[0, :2], num_target)  # center      # Assign Center point for inital reach of the trial (before delay)
            # Assign first target to input for delay period
            input_tensor[n_timesteps:n_timesteps + self.delay_range[1] + self.bump_length] =\
                input_tensor[n_timesteps + self.delay_range[1] + self.bump_length+1, :]
            # append the current trial to temp list, later stacks to tensor with first dimension = batch size
            temp_inputs.append(np.concatenate([input_tensor, np.expand_dims(bump, axis=1)], axis=1))

            # Create targets (outputs)
            # ---------------
            targets[i, :n_timesteps, :] = center  # Reach to Center during delay
            targets[i, n_timesteps:n_timesteps + delay_time, :] = center  # Reach to Center during delay
            # Show True targets after go random bump
            targets[i, n_timesteps + delay_time:n_timesteps + delay_time + int(num_target * n_timesteps), :] =\
                np.concatenate([np.squeeze(target_list[i, :, :, tr]) for tr in range(num_target)], axis=0)
            # Fill the remaining time point at the end with last target (That happens due to random length of bump)
            targets[i, delay_time+(num_target*n_timesteps):, :] = target_list[i, 0, :, -1]

        inputs = tf.stack(temp_inputs, axis=0)
        return [inputs, tf.convert_to_tensor(targets), init_states]


class SequenceHorizon(Task):
    def __init__(self, controller, initial_joint_state=None, **kwargs):
        super().__init__(controller, initial_joint_state=initial_joint_state)
        self.__name__ = 'TaskDelayedMultiReach'
        max_iso_force = self.plant.Muscle.max_iso_force
        self.add_loss('muscle state', loss=ActivationSquaredLoss(max_iso_force=max_iso_force), loss_weight=.2)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)

        self.bump_length = int(kwargs.get('bump_length', 50) / 1000 / self.plant.dt)
        self.bump_height = kwargs.get('bump_height', 3)
        self.delay_range = np.array(kwargs.get('delay_range', [100, 900])) / 1000 / self.plant.dt
        self.num_target = kwargs.get('num_target', 1)
        self.num_horizon = kwargs.get('num_horizon', 0)

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        center = self.plant.joint2cartesian(init_states[0][0, :])

        target_list = np.zeros((batch_size, n_timesteps, 4, self.num_target))

        for tg in range(self.num_target):
            goal_states = self.plant.draw_random_uniform_states(batch_size=batch_size)
            target_list[:, :, :, tg] = self.plant.state2target(
                state=self.plant.joint2cartesian(goal_states),
                n_timesteps=n_timesteps).numpy()

        # Fill in itial part of trials
        # Go to center
        center_in = np.zeros((batch_size, n_timesteps, 2*(self.num_horizon+1)+1))
        center_out = np.zeros((batch_size, n_timesteps, 4))
        # Assign
        center_in[:, :, :2] = center[0, :2]
        center_out[:, :, 0:] = center

        # Show the first n_horizon target and stary in center
        delay_time = generate_delay_time(self.delay_range[0], self.delay_range[1], 'random')

        center_before_go_in = np.zeros((batch_size, delay_time, 2*(self.num_horizon+1)+1))
        center_before_go_out = np.zeros((batch_size, delay_time, 4))
        # Assign
        center_before_go_in[:, :, 0:2*(self.num_horizon+1)] = target_list[:, 0:delay_time, 0:2, 0:self.num_horizon+1].reshape(batch_size, delay_time, -1)
        center_before_go_out[:, :, 0:] = center

        # Get the go-cue bump
        go_in = np.zeros((batch_size, self.bump_length, 2*(self.num_horizon+1)+1))
        go_out = np.zeros((batch_size, self.bump_length, 4))

        go_in[:, :, 0:2*(self.num_horizon+1)] = target_list[:, 0:self.bump_length, 0:2, 0:self.num_horizon+1].reshape(batch_size, self.bump_length, -1)
        go_in[:, :, -1] = np.ones((self.bump_length, )) * self.bump_length
        go_out[:, :, 0:] = center

        # Stack void targets for the end of sequence
        # What should I put at the end?
        target_list = np.concatenate((target_list, np.zeros((batch_size, n_timesteps, 4, self.num_horizon))), axis=-1)

        target_input = np.zeros((batch_size, self.num_target*n_timesteps, 2*(self.num_horizon+1)+1))
        target_output = np.zeros((batch_size, self.num_target*n_timesteps, 4))

        for tg in range(self.num_target):
            target_input[:, tg*n_timesteps:(tg+1)*n_timesteps, 0:2*(self.num_horizon+1)] = target_list[:, :, 0:2, tg:tg+(self.num_horizon+1)].reshape(batch_size, n_timesteps, -1)
            target_output[:, tg*n_timesteps:(tg+1)*n_timesteps, :] = target_list[:, :, :, tg]
        # COncatenate different part of the task
        inp = np.concatenate((center_in, center_before_go_in, go_in, target_input), axis=1)
        outp = np.concatenate((center_out, center_before_go_out, go_out, target_output), axis=1)

        return [tf.convert_to_tensor(inp), tf.convert_to_tensor(outp), init_states]


class TaskLoadProbabilityDistributed(Task):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskLoadProbabilityDistributed'
        self.cartesian_loss = kwargs.get('cartesian_loss', 1.)
        self.muscle_loss = kwargs.get('muscle_loss', 0.)
        self.vel_weight = kwargs.get('vel_weight', 0.)
        self.activity_weight = kwargs.get('activity_weight', 0.)
        self.recurrent_weight = kwargs.get('recurrent_weight', 0.)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=self.cartesian_loss)
        max_iso_force = self.plant.Muscle.max_iso_force
        self.add_loss('muscle state',
                      loss=ActivationVelocitySquaredLoss(max_iso_force=max_iso_force,
                                                         muscle_loss=self.muscle_loss,
                                                         vel_weight=self.vel_weight),
                      loss_weight=1.)
        self.add_loss(assigned_output='gru_hidden0',
                      loss=RecurrentActivityRegularizer(self.controller,
                                                        activity_weight=self.activity_weight,
                                                        recurrent_weight=self.recurrent_weight),
                      loss_weight=1.)
        pre_range = np.array(kwargs.get('pre_range', [0.3, 0.3])) / self.plant.dt
        self.pre_range = [int(pre_range[0]), int(pre_range[1])]
        c1_range = np.array(kwargs.get('c1_range', [0.6, 1.0])) / self.plant.dt
        self.c1_range = [int(c1_range[0]), int(c1_range[1])]
        c2_range = np.array(kwargs.get('c2_range', [0.6, 1.0])) / self.plant.dt
        self.c2_range = [int(c2_range[0]), int(c2_range[1])]
        self.size_range = np.array(kwargs.get('size_range', [0., 0.1]))
        self.target_size = np.array(kwargs.get('target_size', 0.05))
        self.background_load = kwargs.get('background_load', 0.)
        self.run_mode = kwargs.get('run_mode', 'train')
        self.do_recompute_targets = kwargs.get('do_recompute_targets', False)
        self.delay_range = [500, 500]  # this has to exist to get the size of the inputs

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        center_joint = init_states[0][0, :]
        center = self.plant.joint2cartesian(center_joint).numpy()
        # these are our target locations for the experiment version
        goal_state1 = self.plant.joint2cartesian(center_joint + np.deg2rad([-2, 12, 0, 0])).numpy()
        goal_state2 = self.plant.joint2cartesian(center_joint + np.deg2rad([2, -12.3, 0, 0])).numpy()
        goal_states = self.plant.joint2cartesian(self.plant.draw_random_uniform_states(batch_size=batch_size)).numpy()
        targets = np.tile(np.expand_dims(goal_states, axis=1), (1, n_timesteps, 1))
        #pos_pert_ind = 21
        #neg_pert_ind = 8
        pos_pert_ind = 24
        neg_pert_ind = 5
        prob_array = np.array([0, 0.25, 0.5, 0.75, 1])
        proprioceptive_delay = self.controller.proprioceptive_delay
        inputs = np.zeros(shape=(batch_size, n_timesteps, 30 + 2 + 1))
        perturbations = np.zeros(shape=(batch_size, n_timesteps, 2))
        #pert_range = np.linspace(-0.3346, 0.3346, 30)
        pert_range = np.linspace(-1.5263, 1.5263, 30) * 3 / 4
        visual_delay = 7
        if self.run_mode == 'experiment':
            catch_chance = 0.
            pre_range = np.array([0.3, 0.3]) / self.plant.dt
            self.pre_range = [int(pre_range[0]), int(pre_range[1])]
            c1_range = np.array([0.8, 0.8]) / self.plant.dt
            self.c1_range = [int(c1_range[0]), int(c1_range[1])]
            c2_range = np.array([0.8, 0.8]) / self.plant.dt
            self.c2_range = [int(c2_range[0]), int(c2_range[1])]
            self.target_size = np.array([0.05])
        else:
            catch_chance = 0.2  # 0.2 best
        for i in range(batch_size):
            c1_time = generate_delay_time(self.pre_range[0], self.pre_range[1], 'random')
            c2_time = c1_time + generate_delay_time(self.c1_range[0], self.c1_range[1], 'random')
            pert_time = c2_time + generate_delay_time(self.c2_range[0], self.c2_range[1], 'random')
            target_size = np.random.uniform(self.size_range[0], self.size_range[1])
            # randomize cue order
            if np.greater_equal(np.random.rand(), 0.5):
                targ_time = c1_time
                prob_time = c2_time
            else:
                targ_time = c2_time
                prob_time = c1_time
            # is this a catch trial?
            if np.greater_equal(np.random.rand(), catch_chance):
                is_catch = False
            else:
                is_catch = True
            if self.run_mode == 'experiment' or self.run_mode == 'train_experiment':
                # Inputs
                prob = prob_array[np.random.randint(0, 5)]
                inputs[i, prob_time + visual_delay:, pos_pert_ind] = prob
                inputs[i, prob_time + visual_delay:, neg_pert_ind] = 1 - prob
                elb_prob = inputs[i, prob_time + visual_delay, 0:30].copy()
                target_size = self.target_size
                # The following is our expectation violation condition
                if self.run_mode == 'experiment':
                    pos_prob_condition = inputs[i, -1, pos_pert_ind].copy()
                    if pos_prob_condition == 0 or pos_prob_condition == 1:
                        if np.random.rand() < 0.5:
                            elb_prob[pos_pert_ind] = 1 - prob
                            elb_prob[neg_pert_ind] = prob
                # Targets
                if np.random.rand() < 0.5:
                    targets[i, :, :] = np.tile(np.expand_dims(goal_state1, axis=1), [1, n_timesteps, 1])
                else:
                    targets[i, :, :] = np.tile(np.expand_dims(goal_state2, axis=1), [1, n_timesteps, 1])
            elif self.run_mode == 'train':
                prob = prob_array[np.random.randint(0, 5)]
                pert_ind_1 = 0
                pert_ind_2 = 0
                while pert_ind_1 == pert_ind_2:
                    [pert_ind_1, pert_ind_2] = np.random.randint(0, high=30, size=2)
                inputs[i, prob_time + visual_delay:, pert_ind_1] = prob
                inputs[i, prob_time + visual_delay:, pert_ind_2] = 1 - prob
                elb_prob = inputs[i, prob_time + visual_delay, 0:30].copy()
                # Targets
                r = 0.1 * np.sqrt(np.random.rand())
                theta = np.random.rand() * 2 * np.pi
                new_pos = center[0, 0:2] + np.expand_dims([r * np.cos(theta), r * np.sin(theta)], axis=0)
                targets[i, :, 0:2] = np.tile(np.expand_dims(new_pos, axis=1), [1, n_timesteps, 1])

            inputs[i, targ_time + visual_delay:, 30:32] = targets[i, targ_time + visual_delay:, 0:2]
            perturbation = np.tile([0., self.background_load], (n_timesteps, 1))  # background load
            if not is_catch:
                targets[i, 0:pert_time, :] = center
                big_prob = []
                for j in range(len(elb_prob)):
                    big_prob.append(np.tile(j, int(np.round(elb_prob[j]*100))))
                big_prob = [item for sublist in big_prob for item in sublist]
                perturbation[pert_time:, 1] = self.background_load + pert_range[random.choice(big_prob)]
            else:
                targets[i, :, :] = center

            # let's tell the network how big the target is
            inputs[i, targ_time + visual_delay:, 32] = target_size

            #inputs[i, :, :] = inputs[i, :, :] + np.random.normal(loc=0., scale=self.controller.visual_noise_sd,
            #                                                     size=(n_timesteps, 33))
            perturbations[i, :, :] = perturbation

        all_inputs = {"inputs": inputs, "joint_load": perturbations}
        return [all_inputs, tf.convert_to_tensor(targets, dtype=tf.float32), init_states]

    @staticmethod
    def recompute_targets(inputs, targets, outputs):
        # get target size
        target_size = tf.cast(inputs[0]['inputs'][:, -1, 32], tf.float32)
        # get joint loads
        joint_load = inputs[0]["joint_load"]
        # grab endpoint position and velocity
        cartesian_pos = outputs['cartesian position']
        # calculate the distance to the targets as run by the forward pass
        dist = tf.sqrt(tf.reduce_sum((cartesian_pos[:, :, 0:2] - targets[:, :, 0:2])**2, axis=2))
        dist = tf.where(tf.equal(joint_load[:, :, -1], -0.25), 1000., dist)
        dist = tf.tile(tf.expand_dims(dist, axis=2), tf.constant([1, 1, 2], tf.int32))
        cartesian_pos_no_vel = tf.concat([cartesian_pos[:, :, 0:2], tf.zeros_like(dist)], axis=2)
        dist = tf.concat([dist, tf.zeros_like(dist)], axis=2)
        # tile out the target size matrix
        size_matrix = tf.tile(tf.expand_dims(target_size, axis=1), tf.constant([1, dist.shape[1]], tf.int32))
        size_matrix = tf.tile(tf.expand_dims(size_matrix, axis=2), tf.constant([1, 1, 2], tf.int32))
        size_matrix = tf.concat([size_matrix, tf.zeros_like(size_matrix)], axis=2)
        # if the distance is less than a certain amount, replace the target with the result of the forward pass
        targets = tf.where(tf.less_equal(dist, size_matrix), cartesian_pos_no_vel, targets)
        return targets


class TaskYangetal2011(Task):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskYangetal2011'
        max_iso_force = self.plant.Muscle.max_iso_force
        self.add_loss('muscle state', loss=ActivationSquaredLoss(max_iso_force=max_iso_force), loss_weight=0.5)
        self.add_loss('cartesian position', loss=PositionLoss(), loss_weight=1.)  # 2-10 best
        self.do_recompute_targets = True
        self.controller.perturbation_dim_start = 2

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        center = self.plant.joint2cartesian(init_states[0][0, :]).numpy()
        goal_state1 = center + np.array([-0.15551, -0.13049, 0, 0])
        goal_state2 = center + np.array([0.15551, 0.13049, 0, 0])
        delay_array = np.array([2000, 500, 190, 170, 150, 130, 110, 90, 70, 50, 30, 10, -90]) / 1000 / self.plant.dt
        targets = np.tile(center, (batch_size, n_timesteps, 1))
        inputs = np.zeros(shape=(batch_size, n_timesteps, 4))
        for i in range(batch_size):
            input_1 = np.zeros(shape=n_timesteps)
            input_2 = np.zeros(shape=n_timesteps)
            perturbation = np.tile([0, -2], (n_timesteps, 1))  # background load
            target_delay = generate_delay_time(300 / 1000 / self.plant.dt, 700 / 1000 / self.plant.dt, 'random')
            delay_time = int(delay_array[np.random.randint(0, 13)])
            pert_time = target_delay + delay_time
            if np.greater_equal(np.random.rand(),  0.5):
                targets[i, pert_time+1:, :] = goal_state1
                input_1[target_delay + 10:] = 1
            else:
                targets[i, pert_time+1:, :] = goal_state2
                input_2[target_delay + 10:] = 1
            if np.greater_equal(np.random.rand(), 0.5):  # 0.5 best
                perturbation[pert_time:, 1] = -4
            else:
                targets[i, :, :] = center

            inputs[i, :, 0] = input_1
            inputs[i, :, 1] = input_2
            inputs[i, :, 0:2] = inputs[i, :, 0:2] + np.random.normal(loc=0.,
                                                                     scale=0.01,
                                                                     size=(1, n_timesteps, 2))
            inputs[i, :, 2:4] = perturbation

        return [tf.convert_to_tensor(inputs, dtype=tf.float32), tf.convert_to_tensor(targets, dtype=tf.float32),
                init_states]

    @staticmethod
    def recompute_targets(inputs, targets, outputs):
        # grab endpoint position and velocity
        cartesian_pos = outputs['cartesian position']
        # calculate the distance to the targets as run by the forward pass
        dist = tf.sqrt(tf.reduce_sum((cartesian_pos[:, :, 0:2] - targets[:, :, 0:2])**2, axis=2))
        dist = tf.where(tf.equal(inputs[0][:, :, 3], -2.), 1000., dist)
        dist = tf.tile(tf.expand_dims(dist, axis=2), tf.constant([1, 1, 2], tf.int32))
        cartesian_pos_no_vel = tf.concat([cartesian_pos[:, :, 0:2], tf.zeros_like(dist)], axis=2)
        dist = tf.concat([dist, tf.zeros_like(dist)], axis=2)
        # if the distance is less than a certain amount, replace the target with the result of the forward pass
        targets = tf.where(tf.less_equal(dist, 0.2), cartesian_pos_no_vel, targets)
        return targets


def generate_delay_time(delay_min, delay_max, delay_mode):
    if delay_mode == 'random':
        delay_time = np.random.uniform(delay_min, delay_max)
    elif delay_mode == 'noDelayInput':
        delay_time = 0
    else:
        raise AttributeError

    return int(delay_time)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
