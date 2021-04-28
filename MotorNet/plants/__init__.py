import numpy as np
import tensorflow as tf
from MotorNet.plants.skeletons import TwoDofArm
from MotorNet.plants.muscles import RigidTendonHillMuscle, RigidTendonHillMuscleThelen


class Plant:

    def __init__(self, skeleton, timestep=0.01, **kwargs):

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

        kwargs.setdefault('timestep', self.dt)
        for key, val in kwargs.items():
            if key in self.tobuild__muscle:
                self.tobuild__muscle[key].append(val)
        for key, val in self.tobuild__muscle.items():
            if len(val) < self.n_muscles:
                raise ValueError('Missing keyword argument ' + key + '.')
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
        muscle0 = self.Muscle.get_initial_muscle_state(batch_size=batch_size, geometry=geometry0)
        return [joint0, cartesian0, muscle0, geometry0]

    def __call__(self, muscle_input, joint_state, muscle_state, geometry_state, **kwargs):
        endpoint_load = kwargs.get('endpoint_load', np.zeros(1))
        #joint_load = kwargs.get('joint_load', np.zeros(1))
        #joint_load = tf.constant(joint_load, shape=(1, self.Skeleton.dof), dtype=tf.float32)
        joint_load = kwargs.get('joint_load', tf.constant(0., shape=(1, self.Skeleton.dof), dtype=tf.float32))

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

    def __init__(self, timestep=0.01, **kwargs):

        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])

        super().__init__(
            skeleton=TwoDofArm(m1=2.10, m2=1.65, L1g=.146, L2g=.179, I1=.024, I2=.025, L1=.335, L2=.263),
            muscle_type=RigidTendonHillMuscle(),
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
        self.tobuild__muscle = self.Muscle.to_build_dict
        self.Muscle.build(
            timestep=0.01,
            max_isometric_force=[838, 1207, 1422, 1549, 414, 603],
            tendon_length=[0.069, 0.066, 0.172, 0.187, 0.204, 0.217],
            optimal_muscle_length=[0.134, 0.140, 0.092, 0.093, 0.137, 0.127])
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


class RigidTendonArmThelen(PlantWrapper):

    def __init__(self, timestep=0.01, **kwargs):

        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])
        # skeleton = TwoDofArm(m1=2.10, m2=1.65, L1g=.146, L2g=.179, I1=.024, I2=.025, L1=.335, L2=.263),
        skeleton = TwoDofArm(m1=1.82, m2=1.43, L1g=.135, L2g=.165, I1=.051, I2=.057, L1=.309, L2=.333)

        super().__init__(
            skeleton=skeleton,
            muscle_type=RigidTendonHillMuscleThelen(),
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
        self.tobuild__muscle = self.Muscle.to_build_dict
        self.Muscle.build(
            timestep=0.01,
            max_isometric_force=[838, 1207, 1422, 1549, 414, 603],
            tendon_length=[0.039, 0.066, 0.172, 0.187, 0.204, 0.217],
            optimal_muscle_length=[0.134, 0.140, 0.092, 0.093, 0.137, 0.127])
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
        musculotendon_len = tf.reduce_sum(old_pos * self.a1, axis=1, keepdims=True) + \
                            tf.reduce_sum(old_pos2 * self.a2, axis=1, keepdims=True) + self.a0
        musculotendon_vel = tf.reduce_sum(old_vel * moment_arm, axis=1, keepdims=True)
        return tf.concat([musculotendon_len, musculotendon_vel, moment_arm], axis=1)


class RigidTendonArmThelen2(RigidTendonArm):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.l0_pe = self.l0_ce

        # pre-computed for speed
        self.gamma = 0.45
        self.vmax = 10 * self.l0_ce
        self.ce_Af = 0.25
        self.fmlen = 1.4
        self.k_pe = 5.
        self.pe_1 = self.k_pe / 0.6
        self.pe_den = tf.exp(self.k_pe) - 1
        self.ce_0 = 3. * self.vmax
        self.ce_1 = 3. * self.ce_Af * self.vmax * self.fmlen
        self.ce_2 = 3. * self.ce_Af * self.vmax
        self.ce_3 = 8. * self.ce_Af * self.fmlen
        self.ce_4 = self.ce_Af * self.fmlen
        self.ce_5 = self.ce_Af * self.vmax
        self.ce_6 = 8. * self.fmlen
        self.ce_7 = self.ce_Af * self.fmlen * self.vmax

    def __call__(self, excitation, joint_state, muscle_state, geometry_state, endpoint_load=0.):
        # compute muscle activations
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1])
        excitation = tf.reshape(excitation, (-1, 1, self.n_muscles))
        excitation += tf.random.normal(tf.shape(excitation), mean=0., stddev=self.excitation_noise_sd)
        activation = tf.clip_by_value(activation, self.min_activation, 1.)
        excitation = tf.clip_by_value(excitation, self.min_activation, 1.)

        tau_scaler = 0.5 + 1.5 * activation
        tau = tf.where(excitation > activation, self.tau_activation * tau_scaler, self.tau_deactivation / tau_scaler)
        d_activation = (excitation - activation) / tau
        new_activation = activation + d_activation * self.dt
        new_activation = tf.clip_by_value(new_activation, self.min_activation, 1.)

        musculotendon_len, musculotendon_vel, moment_arm = self.get_geometry(joint_state)
        muscle_len = tf.maximum((musculotendon_len - self.l0_se) / self.l0_ce, 0.001)
        muscle_vel = musculotendon_vel

        # muscle forces
        new_activation3 = new_activation * 3.
        nom = tf.where(
            condition=muscle_vel <= 0,
            x=self.ce_Af * (new_activation * self.ce_0 + 4. * muscle_vel + self.vmax),
            y=(self.ce_1 * new_activation
               - self.ce_2 * new_activation
               + self.ce_3 * muscle_vel
               + self.ce_7
               - self.ce_5
               + self.ce_6 * muscle_vel))
        den = tf.where(
            condition=muscle_vel <= 0,
            x=(new_activation3 * self.ce_5 + self.ce_5 - 4. * muscle_vel),
            y=(new_activation3 * self.ce_7
               - new_activation3 * self.ce_5
               + self.ce_7
               + 8. * self.ce_Af * muscle_vel
               - self.ce_5
               + 8. * muscle_vel))
        fvce = tf.maximum(nom / den, 0.)
        flpe = (tf.exp(self.pe_1 * (muscle_len - self.l0_pe / self.l0_ce)) - 1) / self.pe_den
        flce = tf.exp((- (muscle_len - 1) ** 2) / self.gamma)
        f = (new_activation * flce * fvce + flpe) * self.max_iso_force

        generalized_forces = tf.reduce_sum(- f * moment_arm, axis=-1)
        new_joint_state = self.Skeleton(generalized_forces, joint_state, endpoint_load=endpoint_load)
        new_cartesian_state = self.Skeleton.joint2cartesian(joint_state=new_joint_state)

        new_muscle_state = tf.concat(
            [
                new_activation,
                muscle_len * self.l0_ce,
                muscle_vel,
            ], axis=1)

        new_geometry_state = tf.concat(
            [
                musculotendon_len,
                musculotendon_vel,
                moment_arm
            ], axis=1)

        return new_joint_state, new_cartesian_state, new_muscle_state, new_geometry_state


class CompliantTendonArm(Plant):

    def __init__(self, timestep=0.0001, min_activation=0.01, **kwargs):

        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])

        super().__init__(
            skeleton=TwoDofArm(m1=2.10, m2=1.65, L1g=.146, L2g=.179, I1=.024, I2=.025, L1=.335, L2=.263),
            timestep=timestep,
            pos_lower_bound=(sho_limit[0], elb_limit[0]),
            pos_upper_bound=(sho_limit[1], elb_limit[1]),
            vel_lower_bound=-1e6,
            vel_upper_bound=+1e6,
            **kwargs)

        self.input_dim = 6  # 6 muscles
        self.muscle_state_dim = 6
        self.geometry_state_dim = 4
        self.n_muscles = 6

        self.min_activation = min_activation
        self.tau_activation = 0.015
        self.tau_deactivation = 0.05

        self.a0 = tf.constant([0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989], shape=(1, 1, 6), dtype=tf.float32)
        self.a1 = tf.constant([-.03, .03, 0, 0, -.03, .03,
                               0, 0, -.014, .025, -.016, .03], shape=(1, 2, 6), dtype=tf.float32)
        self.a2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, -4, -2.2, -5.7, -3.2], shape=(1, 2, 6), dtype=tf.float32) * 0.001
        self.l0_se = tf.constant([0.069, 0.066, 0.172, 0.187, 0.204, 0.217], shape=(1, 1, 6), dtype=tf.float32)
        self.l0_ce = tf.constant([0.134, 0.140, 0.092, 0.093, 0.137, 0.127], shape=(1, 1, 6), dtype=tf.float32)
        self.l0_pe = self.l0_ce * 1.4
        self.max_iso_force = tf.constant([838, 1207, 1422, 1549, 414, 603], shape=(1, 1, 6), dtype=tf.float32)

        # pre-computed for speed
        self.s_as = 0.001
        self.f_iso_n_den = .66 ** 2
        self.b_rel_st_den = 5e-3 - 0.3
        self.vmax = 10 * self.l0_ce
        self.k_pe = 1 / ((1.66 - self.l0_pe / self.l0_ce) ** 2)
        self.k_se = 1 / (0.04 ** 2)

    def __call__(self, excitation, joint_state, muscle_state, geometry_state, endpoint_load=0.):
        # compute muscle activations
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1])
        excitation = tf.reshape(excitation, (-1, 1, self.n_muscles))
        excitation += tf.random.normal(tf.shape(excitation), mean=0., stddev=self.excitation_noise_sd)
        activation = tf.clip_by_value(activation, self.min_activation, 1.)
        excitation = tf.clip_by_value(excitation, self.min_activation, 1.)

        tau_scaler = 0.5 + 1.5 * activation
        tau = tf.where(excitation > activation, self.tau_activation * tau_scaler, self.tau_deactivation / tau_scaler)
        d_activation = (excitation - activation) / tau
        new_activation = activation + d_activation * self.dt
        new_activation = tf.clip_by_value(new_activation, self.min_activation, 1.)

        # musculotendon geometry
        muscle_len = tf.slice(muscle_state, [0, 1, 0], [-1, 1, -1])
        muscle_vel = tf.slice(muscle_state, [0, 2, 0], [-1, 1, -1])
        norm_muscle_len = muscle_len / self.l0_ce

        musculotendon_len, musculotendon_vel, moment_arm = self.get_geometry(joint_state)
        tendon_len = musculotendon_len - muscle_len
        tendon_strain = tf.maximum((tendon_len - self.l0_se) / self.l0_se, 0.)
        muscle_strain = tf.maximum((muscle_len - self.l0_pe) / self.l0_ce, 0.)

        # forces
        flse = tf.minimum(self.k_se * (tendon_strain ** 2), 3.)
        flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 3.)
        active_force = tf.maximum(flse - flpe, 0.)

        # RK4 integration
        muscle_vel_k1 = self.muscle_ode(norm_muscle_len, new_activation, active_force)
        muscle_vel_k2 = self.muscle_ode(norm_muscle_len + self.dt * 0.5 * muscle_vel_k1, new_activation, active_force)
        muscle_vel_k3 = self.muscle_ode(norm_muscle_len + self.dt * 0.5 * muscle_vel_k2, new_activation, active_force)
        muscle_vel_k4 = self.muscle_ode(norm_muscle_len + self.dt * muscle_vel_k3, new_activation, active_force)
        new_muscle_vel = (muscle_vel_k1 + 2 * muscle_vel_k2 + 2 * muscle_vel_k3 + muscle_vel_k4) / 6
        new_muscle_len = (norm_muscle_len + self.dt * (new_muscle_vel - 0.0 * muscle_vel / self.vmax)) * self.l0_ce

        generalized_forces = - tf.reduce_sum(flse * self.max_iso_force * moment_arm, axis=-1)
        new_joint_state = self.Skeleton(generalized_forces, joint_state, endpoint_load=endpoint_load)
        new_cartesian_state = self.Skeleton.joint2cartesian(joint_state=new_joint_state)

        new_muscle_state = tf.concat(
            [
                new_activation,
                new_muscle_len,
                new_muscle_vel * self.vmax,
                flpe,
                flse,
                active_force
            ], axis=1)

        new_geometry_state = tf.concat(
            [
                musculotendon_len,
                musculotendon_vel,
                moment_arm
            ], axis=1)

        return new_joint_state, new_cartesian_state, new_muscle_state, new_geometry_state

    def muscle_ode(self, norm_muscle_len, activation, active_force):
        f_iso_n = 1 + (- norm_muscle_len ** 2 + 2 * norm_muscle_len - 1) / self.f_iso_n_den
        f_iso_n = tf.maximum(f_iso_n, 0.01)

        a_rel_st = tf.where(norm_muscle_len > 1., .41 * f_iso_n, .41)
        b_rel_st = tf.where(activation < 0.3, 5.2 * (1 - .9 * ((activation - 0.3) / self.b_rel_st_den)) ** 2, 5.2)

        dvdf_isom_con = b_rel_st / (activation * (f_iso_n + a_rel_st))  # slope at isometric point wrt concentric curve
        dfdvcon0 = 1. / dvdf_isom_con
        f_x_a = f_iso_n * activation  # to speed up computation

        p1 = -(f_x_a * 0.5) / (self.s_as - dfdvcon0 * 2)
        p3 = - 1.5 * f_x_a
        p4 = - self.s_as
        # defensive code to ensure this p2 never explode to inf (this way p2 is divided before it is multiplied)
        p2_containing_term = (4 * ((f_x_a * 0.5) ** 2) * p4) / (self.s_as - dfdvcon0 * 2)

        sqrt_term = active_force ** 2 - 2 * active_force * p1 * p4 + \
            2 * active_force * p3 + p1 ** 2 * p4 ** 2 - 2 * p1 * p3 * p4 + p2_containing_term + p3 ** 2

        # defensive code to avoid propagation of negative square root in the non-selected tf.where outcome
        # the assertion is to ensure that any negative root is indeed a non-selected item.
        cond = tf.where(tf.logical_and(sqrt_term < 0, active_force >= f_x_a), -1, 1)
        tf.debugging.assert_non_negative(cond, message='root that should be used is negative.')
        sqrt_term = tf.maximum(sqrt_term, 0.)

        new_muscle_vel_nom = tf.where(
            condition=active_force < f_x_a,
            x=b_rel_st * (active_force - f_x_a),
            y=-active_force + p1 * self.s_as - p3 - tf.sqrt(sqrt_term))
        new_muscle_vel_den = tf.where(
            condition=active_force < f_x_a,
            x=active_force + activation * a_rel_st,
            y=-2 * self.s_as)

        return new_muscle_vel_nom / new_muscle_vel_den

    def get_geometry(self, joint_state):
        old_pos, old_vel = tf.split(joint_state[:, :, tf.newaxis], 2, axis=1)
        old_pos = old_pos - np.array([np.pi / 2, 0.]).reshape((1, 2, 1))
        old_pos2 = tf.pow(old_pos, 2)

        musculotendon_len = tf.reduce_sum(old_pos * self.a1 + old_pos2 * self.a2, axis=1, keepdims=True) + self.a0
        moments = old_pos * self.a2 * 2 + self.a1
        musculotendon_vel = tf.reduce_sum(old_vel * moments, axis=1, keepdims=True)

        # these outputs are **NOT** normalized
        # variables are not concatenated here for speed
        return musculotendon_len, musculotendon_vel, moments

    def get_initial_state(self, batch_size=1, joint_state=None):
        joint0 = self.parse_initial_joint_state(joint_state=joint_state, batch_size=batch_size)
        cartesian0 = self.Skeleton.joint2cartesian(joint_state=joint0)
        musculotendon_len, musculotendon_vel, moment_arm = self.get_geometry(joint0)
        activation = tf.ones_like(musculotendon_len) * self.min_activation

        kpe = self.k_pe
        kse = self.k_se
        lpe = self.l0_pe
        lse = self.l0_se
        lce = self.l0_ce
        lmt = musculotendon_len

        # if musculotendon length is negative, raise an error.
        # if musculotendon length is less than tendon slack length, assign all (most of) the length to the tendon.
        # if musculotendon length is more than tendon slack length and less than muscle slack length, assign to the
        #   tendon up to the tendon slack length, and the rest to the muscle length.
        # if musculotendon length is more than tendon slack length and muscle slack length combined, find the muscle
        #   length that satisfies equilibrium between tendon passive forces and muscle passive forces.
        muscle_len = tf.where(
            musculotendon_len < 0,
            -1,
            tf.where(
                musculotendon_len < lse,
                0.001 * self.l0_ce,
                tf.where(
                    musculotendon_len < lse + lpe,
                    musculotendon_len - lse,
                    (kpe * lpe * lse ** 2 - kse * lce ** 2 * lmt + kse * lce ** 2 * lse - lce * lse * tf.sqrt(kpe * kse)
                     * (-lmt + lpe + lse)) / (kpe * lse ** 2 - kse * lce ** 2)
                )
            )
        )
        tf.debugging.assert_non_negative(muscle_len, message='initial muscle length was < 0.')
        z = tf.zeros_like(muscle_len)
        muscle0 = tf.concat((activation, muscle_len, musculotendon_vel, z, z, z), axis=1)
        geometry0 = tf.concat((musculotendon_len, musculotendon_vel, moment_arm), axis=1)
        return [joint0, cartesian0, muscle0, geometry0]
