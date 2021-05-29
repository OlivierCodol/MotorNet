import numpy as np
import tensorflow as tf
from MotorNet.plants.skeletons import TwoDofArm
from MotorNet.plants.muscles import CompliantTendonHillMuscle


class Plant:

    def __init__(self, skeleton, timestep=0.01, **kwargs):
        self.__name__ = 'Plant'
        self.Skeleton = skeleton
        self.dof = self.Skeleton.dof
        self.space_dim = self.Skeleton.space_dim
        self.state_dim = self.Skeleton.state_dim
        self.output_dim = self.Skeleton.output_dim
        self.dt = timestep
        self.excitation_noise_sd = kwargs.get('excitation_noise_sd', 0.)

        # default is no delay
        proprioceptive_delay = kwargs.get('proprioceptive_delay', self.dt)
        visual_delay = kwargs.get('visual_delay', self.dt)
        self.proprioceptive_delay = int(proprioceptive_delay / self.dt)
        self.visual_delay = int(visual_delay / self.dt)

        # handle position & velocity ranges
        pos_lower_bound = kwargs.get('pos_lower_bound', self.Skeleton.pos_lower_bound)
        pos_upper_bound = kwargs.get('pos_upper_bound', self.Skeleton.pos_upper_bound)
        vel_lower_bound = kwargs.get('vel_lower_bound', self.Skeleton.vel_lower_bound)
        vel_upper_bound = kwargs.get('vel_upper_bound', self.Skeleton.vel_upper_bound)
        pos_bounds = self.set_state_limit_bounds(lb=pos_lower_bound, ub=pos_upper_bound)
        vel_bounds = self.set_state_limit_bounds(lb=vel_lower_bound, ub=vel_upper_bound)
        self.pos_upper_bound = tf.constant(pos_bounds[:, 1], dtype=tf.float32)
        self.pos_lower_bound = tf.constant(pos_bounds[:, 0], dtype=tf.float32)
        self.vel_upper_bound = tf.constant(vel_bounds[:, 1], dtype=tf.float32)
        self.vel_lower_bound = tf.constant(vel_bounds[:, 0], dtype=tf.float32)

        self.Skeleton.build(
            timestep=self.dt,
            pos_upper_bound=self.pos_upper_bound,
            pos_lower_bound=self.pos_lower_bound,
            vel_upper_bound=self.vel_upper_bound,
            vel_lower_bound=self.vel_lower_bound)

        self.built = False

    def draw_random_uniform_states(self, batch_size=1):
        # create a batch of new targets in the correct format for the tensorflow compiler
        sz = (batch_size, self.dof)
        lo = self.pos_lower_bound
        hi = self.pos_upper_bound
        pos = tf.random.uniform(sz, lo, hi)
        vel = tf.zeros(sz)
        return tf.concat([pos, vel], axis=1)

    def parse_initial_joint_state(self, joint_state, batch_size=1):
        if joint_state is None:
            joint0 = self.draw_random_uniform_states(batch_size=batch_size)
        else:
            if tf.shape(joint_state)[0] > 1:
                batch_size = 1
            n_state = joint_state.shape[1]
            if n_state == self.state_dim:
                position, velocity = tf.split(joint_state, 2, axis=-1)
                joint0 = self.draw_fixed_states(position=position, velocity=velocity, batch_size=batch_size)
            elif n_state == int(self.state_dim / 2):
                joint0 = self.draw_fixed_states(position=joint_state, batch_size=batch_size)
            else:
                raise ValueError

        return joint0

    def draw_fixed_states(self, position, velocity=None, batch_size=1):
        if velocity is None:
            velocity = np.zeros_like(position)
        # in case input is a list, a numpy array or a tensorflow array
        pos = np.array(position)
        vel = np.array(velocity)
        if len(pos.shape) == 1:
            pos = pos.reshape((1, -1))
        if len(vel.shape) == 1:
            vel = vel.reshape((1, -1))
        assert pos.shape == vel.shape
        assert pos.shape[1] == self.dof
        assert len(pos.shape) == 2
        assert np.all(pos >= self.pos_lower_bound)
        assert np.all(pos <= self.pos_upper_bound)
        assert np.all(vel >= self.vel_lower_bound)
        assert np.all(vel <= self.vel_upper_bound)

        pos = tf.cast(pos, dtype=tf.float32)
        vel = tf.cast(vel, dtype=tf.float32)
        states = tf.concat([pos, vel], axis=1)
        tiled_states = tf.tile(states, [batch_size, 1])
        return tiled_states

    def set_state_limit_bounds(self, lb, ub):
        lb = np.array(lb).reshape((-1, 1))  # ensure this is a 2D array
        ub = np.array(ub).reshape((-1, 1))
        bounds = np.hstack((lb, ub))
        bounds = bounds * np.ones((self.dof, 2))  # if one bound pair inputed, broadcast to dof rows
        return bounds

    def setattr(self, name: str, value):
        self.__setattr__(name, value)

    @staticmethod
    def state2target(state, n_timesteps=1):
        # convert a state array to a target array
        targ = state[:, :, tf.newaxis]
        targ = tf.repeat(targ, n_timesteps, axis=-1)
        targ = tf.transpose(targ, [0, 2, 1])
        return targ

    def get_save_config(self):
        cfg = {'Muscle': str(self.Muscle.__name__),
               'Skeleton': {'I1': self.Skeleton.I1, 'I2': self.Skeleton.I2, 'L1': self.Skeleton.L1,
                            'L2': self.Skeleton.L2, 'L1g': self.Skeleton.L1g, 'L2g': self.Skeleton.L2g,
                            'c_viscosity': self.Skeleton.c_viscosity, 'coriolis_1': self.Skeleton.coriolis_1,
                            'coriolis_2': self.Skeleton.coriolis_2, 'dof': self.Skeleton.dof, 'dt': self.Skeleton.dt,
                            'm1': self.Skeleton.m1, 'm2': self.Skeleton.m2},
               'excitation_noise_sd': self.excitation_noise_sd, 'n_muscles': self.n_muscles,
               'proprioceptive_delay': self.proprioceptive_delay, 'visual_delay': self.visual_delay}
        return cfg

    def get_geometry(self, joint_state):
        pass

    def joint2cartesian(self, joint_state):
        return self.Skeleton.joint2cartesian(joint_state=joint_state)


class PlantWrapper(Plant):

    def __init__(self, skeleton, muscle_type, timestep=0.01, **kwargs):

        super().__init__(skeleton=skeleton, timestep=timestep, **kwargs)

        # initialize muscle system
        self.Muscle = muscle_type
        self.MusclePaths = []  # a list of all the muscle paths
        self.n_muscles = 0
        self.input_dim = 0
        self.muscle_name = []
        self.muscle_state_dim = self.Muscle.state_dim
        self.geometry_state_dim = 2 + self.Skeleton.dof  # musculotendon length & velocity + as many moments as dofs
        self.path_fixation_body = np.empty((1, 1, 0)).astype('float32')
        self.path_coordinates = np.empty((1, self.Skeleton.space_dim, 0)).astype('float32')
        self.muscle = np.empty(0).astype('float32')
        self.tobuild__muscle = self.Muscle.to_build_dict
        self.tobuild__default = self.Muscle.to_build_dict_default
        self.muscle_transitions = None
        self.row_splits = None
        self.built = False

    def add_muscle(self, path_fixation_body: list, path_coordinates: list, name='', **kwargs):
        path_fixation_body = np.array(path_fixation_body).astype('float32').reshape((1, 1, -1))
        n_points = path_fixation_body.size
        path_coordinates = np.array(path_coordinates).astype('float32').T[np.newaxis, :, :]
        assert path_coordinates.shape[1] == self.Skeleton.space_dim
        assert path_coordinates.shape[2] == n_points
        self.n_muscles += 1
        self.input_dim += self.Muscle.input_dim

        # path segments & coordinates should be a (batch_size * n_coordinates  * n_segments * (n_muscles * n_points)
        self.path_fixation_body = np.concatenate([self.path_fixation_body, path_fixation_body], axis=-1)
        self.path_coordinates = np.concatenate([self.path_coordinates, path_coordinates], axis=-1)
        self.muscle = np.hstack([self.muscle, np.tile(np.max(self.n_muscles), [n_points])])

        # indexes where the next item is from a different muscle, to indicate when their difference is meaningless
        self.muscle_transitions = np.diff(self.muscle.reshape((1, 1, -1))) == 1.
        # to create the ragged tensors when collapsing each muscle's segment values
        n_total_points = np.array([len(self.muscle)])
        self.row_splits = np.concatenate([np.zeros(1), np.diff(self.muscle).nonzero()[0] + 1, n_total_points - 1])

        # to prevent raising a "Missing keyword argument" error in the loops below
        kwargs.setdefault('timestep', self.dt)
        # kwargs loop
        for key, val in kwargs.items():
            if key in self.tobuild__muscle:
                self.tobuild__muscle[key].append(val)
        for key, val in self.tobuild__muscle.items():
            # if not added in the kwargs loop
            if len(val) < self.n_muscles:
                # if the muscle object contains a default, use it
                if key in self.tobuild__default:
                    self.tobuild__muscle[key].append(self.tobuild__default[key])
                # else, raise error
                else:
                    raise ValueError('Missing keyword argument ' + key + '.')
        self.tobuild__muscle['timestep'] = [self.dt]
        self.Muscle.build(**self.tobuild__muscle)

        if name == '':
            name = 'muscle_' + str(self.n_muscles)
        self.muscle_name.append(name)

    def get_geometry(self, joint_state):
        xy, dxy_dt, dxy_ddof = self.Skeleton.path2cartesian(self.path_coordinates, self.path_fixation_body, joint_state)
        diff_pos = xy[:, :, 1:] - xy[:, :, :-1]
        diff_vel = dxy_dt[:, :, 1:] - dxy_dt[:, :, :-1]
        diff_ddof = dxy_ddof[:, :, :, 1:] - dxy_ddof[:, :, :, :-1]

        # length, velocity and moment of each path segment
        # -----------------------
        # segment length is just the euclidian distance between the two points
        segment_len = tf.sqrt(tf.reduce_sum(diff_pos ** 2, axis=1, keepdims=True))
        # segment velocity is trickier: we are not after radial velocity but relative velocity.
        # https://math.stackexchange.com/questions/1481701/time-derivative-of-the-distance-between-2-points-moving-over-time
        # Formally, if segment_len=0 then segment_vel is not defined. We could substitute with 0 here because a
        # muscle segment will never flip backward, so the velocity can only be positive afterwards anyway.
        # segment_vel = tf.where(segment_len == 0, tf.zeros(1), segment_vel)
        segment_vel = tf.reduce_sum(diff_pos * diff_vel / segment_len, axis=1, keepdims=True)
        segment_moments = tf.reduce_sum(diff_ddof * diff_pos[:, :, tf.newaxis], axis=1) / segment_len

        # remove differences between points that don't belong to the same muscle
        segment_len_cleaned = tf.where(self.muscle_transitions, 0., segment_len)
        segment_vel_cleaned = tf.where(self.muscle_transitions, 0., segment_vel)
        segment_mom_cleaned = tf.where(self.muscle_transitions, 0., segment_moments)

        # flip all dimensions to allow making ragged tensors below (you need to do it from the rows)
        segment_len_flipped = tf.transpose(segment_len_cleaned, [2, 1, 0])
        segment_vel_flipped = tf.transpose(segment_vel_cleaned, [2, 1, 0])
        segment_mom_flipped = tf.transpose(segment_mom_cleaned, [2, 1, 0])

        # create ragged tensors, which allows to hold each individual muscle's fixation points on the second dimension
        # in case there is not the same number of fixation point for each muscles
        segment_len_ragged = tf.RaggedTensor.from_row_splits(segment_len_flipped, row_splits=self.row_splits)
        segment_vel_ragged = tf.RaggedTensor.from_row_splits(segment_vel_flipped, row_splits=self.row_splits)
        segment_mom_ragged = tf.RaggedTensor.from_row_splits(segment_mom_flipped, row_splits=self.row_splits)

        # now we can sum all segments' contribution along the ragged dimension, that is along all fixation points for
        # each muscle
        musculotendon_len = tf.reduce_sum(segment_len_ragged, axis=1)
        musculotendon_vel = tf.reduce_sum(segment_vel_ragged, axis=1)
        moments = tf.reduce_sum(segment_mom_ragged, axis=1)

        # pack all this into one state array and flip the dimensions back (batch_size * n_features * n_muscles)
        geometry_state = tf.transpose(tf.concat([musculotendon_len, musculotendon_vel, moments], axis=1), [2, 1, 0])
        return geometry_state

    def get_initial_state(self, batch_size=1, joint_state=None):
        if joint_state is not None and tf.shape(joint_state)[0] > 1:
            batch_size = tf.shape(joint_state)[0]
        joint0 = self.parse_initial_joint_state(joint_state=joint_state, batch_size=batch_size)
        cartesian0 = self.Skeleton.joint2cartesian(joint_state=joint0)
        geometry0 = self.get_geometry(joint_state=joint0)
        muscle0 = self.Muscle.get_initial_muscle_state(batch_size=batch_size, geometry_state=geometry0)
        return [joint0, cartesian0, muscle0, geometry0]

    def __call__(self, muscle_input, joint_state, muscle_state, geometry_state, **kwargs):
        endpoint_load = kwargs.get('endpoint_load', np.zeros(1))
        joint_load = kwargs.get('joint_load', tf.constant(0., shape=(1, self.Skeleton.dof), dtype=tf.float32))

        muscle_input += tf.random.normal(tf.shape(muscle_input), stddev=self.excitation_noise_sd)
        forces, new_muscle_state = self.Muscle(excitation=muscle_input,
                                               muscle_state=muscle_state,
                                               geometry_state=geometry_state)
        moments = tf.slice(geometry_state, [0, 2, 0], [-1, -1, -1])
        generalized_forces = - tf.reduce_sum(forces * moments, axis=-1) + joint_load

        new_joint_state = self.Skeleton(generalized_forces, joint_state, endpoint_load=endpoint_load)
        new_cartesian_state = self.Skeleton.joint2cartesian(joint_state=new_joint_state)
        new_geometry_state = self.get_geometry(joint_state=new_joint_state)
        return new_joint_state, new_cartesian_state, new_muscle_state, new_geometry_state


class RigidTendonArm(PlantWrapper):

    def __init__(self, muscle_type, skeleton=None, timestep=0.01, **kwargs):

        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])

        if skeleton is None:
            skeleton = TwoDofArm(m1=1.82, m2=1.43, L1g=.135, L2g=.165, I1=.051, I2=.057, L1=.309, L2=.333)

        super().__init__(
            skeleton=skeleton,
            muscle_type=muscle_type,
            timestep=timestep,
            pos_lower_bound=(sho_limit[0], elb_limit[0]),
            pos_upper_bound=(sho_limit[1], elb_limit[1]),
            **kwargs)

        # build muscle system
        self.muscle_state_dim = self.Muscle.state_dim
        self.geometry_state_dim = 2 + self.Skeleton.dof  # musculotendon length & velocity + as many moments as dofs
        self.n_muscles = 6
        self.input_dim = self.n_muscles
        self.muscle_name = ['pectoralis', 'deltoid', 'brachioradialis', 'tricepslat', 'biceps', 'tricepslong']
        self.Muscle.build(
            timestep=self.dt,
            max_isometric_force=[838, 1207, 1422, 1549, 414, 603],
            tendon_length=[0.039, 0.066, 0.172, 0.187, 0.204, 0.217],
            optimal_muscle_length=[0.134, 0.140, 0.092, 0.093, 0.137, 0.127],
            normalized_slack_muscle_length=self.tobuild__default['normalized_slack_muscle_length'])
        self.built = True

        a0 = [0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989]
        a1 = [-.03, .03, 0, 0, -.03, .03, 0, 0, -.014, .025, -.016, .03]
        a2 = [0, 0, 0, 0, 0, 0, 0, 0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]
        self.a0 = tf.constant(a0, shape=(1, 1, 6), dtype=tf.float32)
        self.a1 = tf.constant(a1, shape=(1, 2, 6), dtype=tf.float32)
        self.a2 = tf.constant(a2, shape=(1, 2, 6), dtype=tf.float32)

    def get_geometry(self, joint_state):
        old_pos, old_vel = tf.split(joint_state[:, :, tf.newaxis], 2, axis=1)
        old_pos = old_pos - np.array([np.pi / 2, 0.]).reshape((1, 2, 1))
        old_pos2 = tf.pow(old_pos, 2)
        moment_arm = old_pos * self.a2 * 2 + self.a1
        musculotendon_len = tf.reduce_sum(old_pos * self.a1 + old_pos2 * self.a2, axis=1, keepdims=True) + self.a0
        musculotendon_vel = tf.reduce_sum(old_vel * moment_arm, axis=1, keepdims=True)
        return tf.concat([musculotendon_len, musculotendon_vel, moment_arm], axis=1)


class CompliantTendonArm(RigidTendonArm):

    def __init__(self, timestep=0.0001, **kwargs):

        super().__init__(
            muscle_type=CompliantTendonHillMuscle(),
            skeleton=TwoDofArm(m1=2.10, m2=1.65, L1g=.146, L2g=.179, I1=.024, I2=.025, L1=.335, L2=.263),
            timestep=timestep,
            **kwargs)

        # build muscle system
        self.Muscle.build(
            timestep=timestep,
            max_isometric_force=[838, 1207, 1422, 1549, 414, 603],
            tendon_length=[0.069, 0.066, 0.172, 0.187, 0.204, 0.217],
            optimal_muscle_length=[0.134, 0.140, 0.092, 0.093, 0.137, 0.127],
            normalized_slack_muscle_length=self.tobuild__default['normalized_slack_muscle_length'])

        a0 = [0.181, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989]
        self.a0 = tf.constant(a0, shape=(1, 1, 6), dtype=tf.float32)
