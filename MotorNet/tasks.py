import numpy as np
from abc import abstractmethod
from MotorNet.nets.losses import position_loss, activation_squared_loss, activation_velocity_squared_loss
import tensorflow as tf


class Task(tf.keras.utils.Sequence):
    def __init__(self, controller, initial_joint_state=None, **kwargs):
        self.__name__ = 'Generic Task'
        self.controller = controller
        self.plant = controller.plant
        self.training_iterations = 1000
        self.training_batch_size = 32
        self.training_n_timesteps = 100
        self.do_recompute_targets = False
        self.kwargs = kwargs

        self.losses = {}
        self.loss_weights = {}

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
        [inputs, _, _] = self.generate(batch_size=1, n_timesteps=5000)
        shape = inputs.get_shape().as_list()
        return shape[-1]

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
        self.losses = {'cartesian position': position_loss(),
                       'muscle state':activation_squared_loss(self.plant.Muscle.max_iso_force)}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.2}  # 0.2

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states = self.plant.draw_random_uniform_states(batch_size=batch_size)
        targets = self.plant.state2target(state=self.plant.joint2cartesian(goal_states), n_timesteps=n_timesteps)
        return [targets[:, :, 0:self.plant.space_dim], targets, init_states]


class TaskStaticTargetWithPerturbations(Task):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskStaticTargetWithPerturbations'
        self.losses = {'cartesian position': position_loss(),
                       'muscle state': activation_squared_loss(self.plant.Muscle.max_iso_force)}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.2}

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states = self.plant.draw_random_uniform_states(batch_size=batch_size)
        targets = self.plant.state2target(state=self.plant.joint2cartesian(goal_states), n_timesteps=n_timesteps)
        perturbations = tf.constant(5., shape=(batch_size, n_timesteps, 2))
        self.controller.perturbation_dims_active = True
        inputs = tf.concat([targets[:, :, 0:self.plant.space_dim], perturbations], axis=2)
        return [inputs, targets, init_states]


class TaskDelayedReach(Task):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskDelayedReach'
        self.losses = {'cartesian position': position_loss(),
                       'muscle state': activation_squared_loss(self.plant.Muscle.max_iso_force)}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.2}

        self.bump_length = int(kwargs.get('bump_length', 50) / 1000 / self.plant.dt)
        self.bump_height = kwargs.get('bump_height', 1)
        self.delay_range = np.array(kwargs.get('delay_range', [100, 900])) / 1000 / self.plant.dt

    def generate(self, batch_size, n_timesteps, **kwargs):
        init_states = self.get_initial_state(batch_size=batch_size)
        center = self.plant.joint2cartesian(init_states[0][0, :])
        goal_states = self.plant.draw_random_uniform_states(batch_size=batch_size)
        targets = self.plant.state2target(state=self.plant.joint2cartesian(goal_states),
                                          n_timesteps=n_timesteps).numpy()

        temp_inputs = []
        for i in range(batch_size):
            delay_time = generate_delay_time(self.delay_range[0], self.delay_range[1], 'random')
            bump = np.concatenate([np.zeros(delay_time), np.ones(self.bump_length)*self.bump_height,
                                   np.zeros(int(n_timesteps - delay_time - self.bump_length))])
            temp_inputs.append(np.concatenate([np.squeeze(targets[i, :, 0:2]), np.expand_dims(bump, axis=1)], axis=1))
            targets[i, 0:delay_time, :] = center

        inputs = tf.stack(temp_inputs, axis=0)
        return [inputs, tf.convert_to_tensor(targets), init_states]


class TaskDelayedMultiReach(Task):
    def __init__(self, controller, initial_joint_state=None, **kwargs):
        super().__init__(controller, initial_joint_state=initial_joint_state)
        self.__name__ = 'TaskDelayedMultiReach'
        self.losses = {'cartesian position': position_loss(),
                       'muscle state': activation_squared_loss(self.plant.Muscle.max_iso_force)}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.2}

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
        self.losses = {'cartesian position': position_loss(),
                       'muscle state': activation_squared_loss(self.plant.Muscle.max_iso_force)}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.2}

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
        center_in[:, : ,:2] = center[0, :2]
        center_out[:,:, 0:] = center  
        
        # Show the first n_horizon target and stary in center
        delay_time = generate_delay_time(self.delay_range[0], self.delay_range[1], 'random')
            
        center_before_go_in =  np.zeros((batch_size, delay_time, 2*(self.num_horizon+1)+1))
        center_before_go_out = np.zeros((batch_size, delay_time, 4)) 
        # Assign
        center_before_go_in[:, : ,0:2*(self.num_horizon+1)] = target_list[:, 0:delay_time, 0:2, 0:self.num_horizon+1].reshape(batch_size, delay_time,-1)
        center_before_go_out[:,:, 0:] = center
        
        # Get the go-cue bump
        go_in =  np.zeros((batch_size, self.bump_length, 2*(self.num_horizon+1)+1))
        go_out = np.zeros((batch_size, self.bump_length, 4))
        
        go_in[:, : ,0:2*(self.num_horizon+1)] = target_list[:, 0:self.bump_length, 0:2, 0:self.num_horizon+1].reshape(batch_size, self.bump_length,-1)
        go_in[:, :, -1] = np.ones((self.bump_length, )) * self.bump_length
        go_out[:, :, 0:] = center
        
                             
        # Stack void targets for the end of sequence 
        # What should I put at the end?
        target_list = np.concatenate((target_list, np.zeros((batch_size, n_timesteps, 4, self.num_horizon))), axis=-1)

        target_input = np.zeros((batch_size,self.num_target*n_timesteps ,2*(self.num_horizon+1)+1 ))
        target_output = np.zeros((batch_size,self.num_target*n_timesteps ,4))
        
        for tg in range(self.num_target):
            target_input[:, tg*n_timesteps:(tg+1)*n_timesteps, 0:2*(self.num_horizon+1)] = target_list[:, :, 0:2, tg:tg+(self.num_horizon+1)].reshape(batch_size, n_timesteps, -1)
            target_output[:, tg*n_timesteps:(tg+1)*n_timesteps, :] = target_list[:, :, :, tg]
        # COncatenate different part of the task
        IN = np.concatenate((center_in, center_before_go_in, go_in, target_input), axis = 1)
        OUT = np.concatenate((center_out, center_before_go_out, go_out, target_output), axis = 1)
        
        return [tf.convert_to_tensor(IN), tf.convert_to_tensor(OUT), init_states]


class TaskLoadProbability(Task):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskLoadProbability'

        self.vel_weight = kwargs.get('vel_weight', 0.)
        self.losses = {'cartesian position': position_loss(),
                       'muscle state': activation_velocity_squared_loss(self.plant.Muscle.max_iso_force,
                                                                        vel_weight=self.vel_weight)}
        self.cartesian_loss = kwargs.get('cartesian_loss', 1)
        self.muscle_loss = kwargs.get('muscle_loss', 0)
        self.loss_weights = {'cartesian position': self.cartesian_loss, 'muscle state': self.muscle_loss}  # 10-20 best

        self.target_time_range = np.array(kwargs.get('target_time_range', [200, 400])) / 1000 / self.plant.dt
        self.delay_range = np.array(kwargs.get('delay_range', [200, 1000])) / 1000 / self.plant.dt
        self.condition_independent_magnitude = kwargs.get('condition_independent_magnitude', 0.2)
        self.background_load = kwargs.get('background_load', 0.)

        self.run_mode = kwargs.get('run_mode', 'train')

        self.do_recompute_targets = kwargs.get('do_recompute_targets', False)
        self.controller.perturbation_dims_active = True

    def generate(self, batch_size, n_timesteps, **kwargs):
        if self.run_mode == 'mesh_target' or self.run_mode == '2target':
            batch_size = 1000
            n_timesteps = 130
            self.delay_range = np.array([500, 500]) / 1000 / self.plant.dt
            self.target_time_range = np.array([200, 200]) / 1000 / self.plant.dt
        init_states = self.get_initial_state(batch_size=batch_size)
        center = self.plant.joint2cartesian(init_states[0][0, :]).numpy()
        goal_state1 = center + np.array([-0.028279, -0.042601, 0, 0])
        goal_state2 = center - np.array([-0.028279, -0.042601, 0, 0])
        goal_states = self.plant.joint2cartesian(self.plant.draw_random_uniform_states(batch_size=batch_size))
        targets = np.tile(np.expand_dims(goal_states, axis=1), (1, n_timesteps, 1))
        prob_array = np.array([0, 0.25, 0.5, 0.75, 1])
        mesh = np.linspace(-0.0707, 0.0707, num=10)
        mesh_x_counter = 0
        mesh_y_counter = 0
        prob_counter = 0
        target_counter = 0
        visual_delay = self.controller.visual_delay
        proprioceptive_delay = self.controller.proprioceptive_delay
        inputs = np.zeros(shape=(batch_size, n_timesteps, 7))
        for i in range(batch_size):
            prob = prob_array[np.random.randint(0, 5)]
            if self.run_mode != 'train':
                prob = prob_array[prob_counter]
                x = mesh[mesh_x_counter]
                y = mesh[mesh_y_counter]
                mesh_x_counter += 1
                new_pos = center[0, 0:2] + np.expand_dims([x, y], axis=0)
                if self.run_mode == 'mesh_target':
                    targets[i, :, 0:2] = np.tile(np.expand_dims(new_pos, axis=1), [1, n_timesteps, 1])
                elif self.run_mode == '2target':
                    if target_counter < 50:
                        targets[i, :, :] = np.tile(np.expand_dims(goal_state1, axis=1), [1, n_timesteps, 1])
                    else:
                        targets[i, :, :] = np.tile(np.expand_dims(goal_state2, axis=1), [1, n_timesteps, 1])
                if mesh_x_counter >= len(mesh):
                    mesh_y_counter += 1
                    mesh_x_counter = 0
                if mesh_y_counter >= len(mesh):
                    mesh_x_counter = 0
                    mesh_y_counter = 0
                    prob_counter += 1
                if prob_counter == 5:
                    prob_counter = 0
                target_counter += 1
                if target_counter >= 100:
                    target_counter = 0
            elif self.run_mode == 'train':
                r = 0.15 * np.sqrt(np.random.rand())
                theta = np.random.rand() * 2 * np.pi
                new_pos = center[0, 0:2] + np.expand_dims([r * np.cos(theta), r * np.sin(theta)], axis=0)
                targets[i, :, 0:2] = np.tile(np.expand_dims(new_pos, axis=1), [1, n_timesteps, 1])
            inputs[i, :, 2:4] = targets[i, :, 0:2]
            input_5 = np.zeros(shape=n_timesteps)
            input_1 = np.zeros(shape=n_timesteps)
            input_2 = np.zeros(shape=n_timesteps)
            perturbation = np.tile([0, self.background_load], (n_timesteps, 1))  # background load
            target_time = generate_delay_time(self.target_time_range[0], self.target_time_range[1], 'random')
            delay_time = generate_delay_time(self.delay_range[0], self.delay_range[1], 'random')
            pert_time = delay_time
            if self.run_mode != 'train':
                catch_chance = 0
                no_prob_chance = 0
            else:
                catch_chance = 0.1  # 0.3 best
                no_prob_chance = 0  # 0 best
            if np.greater_equal(np.random.rand(), catch_chance):
                targets[i, 0:pert_time, :] = center
                if self.run_mode != 'train':
                    if i < 500:
                        pert = self.background_load - 1
                    else:
                        pert = self.background_load + 1
                else:
                    if np.greater_equal(np.random.rand(),  prob):
                        pert = self.background_load + 1
                    else:
                        pert = self.background_load - 1
                if np.greater_equal(np.random.rand(), no_prob_chance):
                    # The visual delay MUST be built into these inputs, otherwise the network gets immediate visual cues
                    input_1[target_time + visual_delay:] = prob  # turn off after 9 best   (pert_time + visual_delay)
                    input_2[target_time + visual_delay:] = 1 - prob
                input_5[pert_time + proprioceptive_delay:] = self.condition_independent_magnitude  # +4 best, 0.2 best
                perturbation[pert_time:, 1] = pert
            else:
                targets[i, :, :] = center
                if np.greater_equal(np.random.rand(), no_prob_chance):
                    input_1[target_time + visual_delay:] = prob
                    input_2[target_time + visual_delay:] = 1 - prob


            inputs[i, :, 0] = input_1
            inputs[i, :, 1] = input_2
            inputs[i, :, 4] = input_5

            inputs[i, :, 4] = inputs[i, :, 4] + np.random.normal(loc=0.,
                                                         scale=self.controller.proprioceptive_noise_sd,
                                                         size=n_timesteps)
            inputs[i, :, 0:4] = inputs[i, :, 0:4] + np.random.normal(loc=0.,
                                                                     scale=self.controller.visual_noise_sd,
                                                                     size=(n_timesteps, 4))
            inputs[i, :, 5:7] = perturbation

        return [tf.convert_to_tensor(inputs, dtype=tf.float32), tf.convert_to_tensor(targets, dtype=tf.float32),
                init_states]

    def recompute_targets(self, inputs, targets, outputs):
        # grab endpoint position and velocity
        cartesian_pos = outputs['cartesian position']
        # calculate the distance to the targets as run by the forward pass
        dist = tf.sqrt(tf.reduce_sum((cartesian_pos[:, :, 0:2] - targets[:, :, 0:2])**2, axis=2))
        dist = tf.where(tf.equal(inputs[0][:, :, -1], -1.), 1000., dist)
        dist = tf.tile(tf.expand_dims(dist, axis=2), tf.constant([1, 1, 2], tf.int32))
        cartesian_pos_no_vel = tf.concat([cartesian_pos[:, :, 0:2], tf.zeros_like(dist)], axis=2)
        dist = tf.concat([dist, tf.zeros_like(dist)], axis=2)
        # if the distance is less than a certain amount, replace the target with the result of the forward pass
        targets = tf.where(tf.less_equal(dist, 0.035), cartesian_pos_no_vel, targets)
        return targets


class TaskYangetal2011(Task):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.__name__ = 'TaskYangetal2011'
        self.losses = {'cartesian position': position_loss(),
                       'muscle state': activation_squared_loss(self.plant.Muscle.max_iso_force)}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.5}  # 2-10 best

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

    def recompute_targets(self, inputs, targets, outputs):
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