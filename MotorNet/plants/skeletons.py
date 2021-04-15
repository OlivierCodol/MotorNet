import numpy as np
import tensorflow as tf


class Arm:
    def __init__(self, timestep=0.01, **kwargs):
        self.dof = 2  # degrees of freedom of the skeleton (eg number of joints)
        self.space_dim = 2  # the dimensionality of the space (eg 2 for cartesian xy space)
        self.input_dim = 6  # 6 muscles
        self.state_dim = self.dof * 2  # usually position and velocity so twice the dof
        self.output_dim = self.state_dim  # usually the new state so same as state_dim
        self.muscle_state_dim = 3
        self.geometry_state_dim = 4
        self.n_muscles = 6
        self.dt = timestep

        # default is no delay
        proprioceptive_delay = kwargs.get('proprioceptive_delay', timestep)
        visual_delay = kwargs.get('visual_delay', timestep)
        self.proprioceptive_delay = int(proprioceptive_delay / self.dt)
        self.visual_delay = int(visual_delay / self.dt)

        self.activation_lower_bound = 0.001
        self.tau_activation = 0.015
        self.tau_deactivation = 0.05

        # handle position & velocity ranges
        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])
        pos_bounds = self.set_state_limit_bounds(lb=(sho_limit[0], elb_limit[0]), ub=(sho_limit[1], elb_limit[1]))
        vel_bounds = self.set_state_limit_bounds(lb=-1e6, ub=1e6)
        self.pos_upper_bounds = tf.constant(pos_bounds[:, 1], dtype=tf.float32)
        self.pos_lower_bounds = tf.constant(pos_bounds[:, 0], dtype=tf.float32)
        self.vel_upper_bounds = tf.constant(vel_bounds[:, 1], dtype=tf.float32)
        self.vel_lower_bounds = tf.constant(vel_bounds[:, 0], dtype=tf.float32)

        a0 = np.array([0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989]).reshape((1, 1, -1))
        a1 = np.array([-0.03, 0.03, 0, 0, -0.03, 0.03, 0, 0, -0.014, 0.025, -0.016, 0.03]).reshape((1, 2, -1))
        a2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, -4, -2.2, -5.7, -3.2]).reshape((1, 2, -1)) * 0.001
        l0_se = np.array([0.039, 0.066, 0.172, 0.187, 0.204, 0.217]).reshape((1, 1, -1))
        l0_ce = np.array([0.134, 0.140, 0.092, 0.093, 0.137, 0.127]).reshape((1, 1, -1))
        l0_pe = l0_ce
        max_iso_force = np.array([838, 1207, 1422, 1549, 414, 603]).reshape((1, 1, -1))

        self.a0 = tf.constant(a0, dtype=tf.float32)
        self.a1 = tf.constant(a1, dtype=tf.float32)
        self.a2 = tf.constant(a2, dtype=tf.float32)
        self.l0_se = tf.constant(l0_se, dtype=tf.float32)
        self.l0_ce = tf.constant(l0_ce, dtype=tf.float32)
        self.l0_pe = tf.constant(l0_pe, dtype=tf.float32)
        self.max_iso_force = tf.constant(max_iso_force, dtype=tf.float32)
        self.min_iso_force = 1.  # in Newtons

        self.m1 = 1.82  # masses of arm links
        self.m2 = 1.43
        self.L1g = 0.135  # center of mass of the links
        self.L2g = 0.165
        self.I1 = 0.051  # moments of inertia around the center of mass
        self.I2 = 0.057
        self.L1 = 0.309  # length of links
        self.L2 = 0.333

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

        # pre-compute values for mass, coriolis, and gravity matrices
        inertia_11_c = self.m1 * self.L1g ** 2 + self.I1 + self.m2 * (self.L2g ** 2 + self.L1 ** 2) + self.I2
        inertia_12_c = self.m2 * (self.L2g ** 2) + self.I2
        inertia_22_c = self.m2 * (self.L2g ** 2) + self.I2
        inertia_11_m = 2 * self.m2 * self.L1 * self.L2g
        inertia_12_m = self.m2 * self.L1 * self.L2g

        # 0-th axis is for broadcasting to batch_size when used
        inertia_c = np.array([[[inertia_11_c, inertia_12_c],
                               [inertia_12_c, inertia_22_c]]]).astype(np.float32)
        inertia_m = np.array([[[inertia_11_m, inertia_12_m],
                               [inertia_12_m, 0.]]]).astype(np.float32)
        self.inertia_c = inertia_c.reshape((1, 2, 2))  # 0-th axis is for broadcasting to batch_size when used
        self.inertia_m = inertia_m.reshape((1, 2, 2))

        self.coriolis_1 = -self.m2 * self.L1 * self.L2g
        self.coriolis_2 = self.m2 * self.L1 * self.L2g
        self.c_viscosity = 0.0  # put at zero but available if implemented later on

    def __call__(self, excitation, joint_state, muscle_state, geometry_state):

        # compute muscle activations
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1])
        excitation = tf.reshape(excitation, (-1, 1, self.n_muscles))
        activation = tf.clip_by_value(activation, self.activation_lower_bound, 1.)
        excitation = tf.clip_by_value(excitation, self.activation_lower_bound, 1.)
        tau_scaler = 0.5 + 1.5 * activation
        tau = tf.where(excitation > activation, self.tau_activation * tau_scaler, self.tau_deactivation / tau_scaler)
        d_activation = (excitation - activation) / tau
        new_activation = activation + d_activation * self.dt
        new_activation = tf.clip_by_value(new_activation, self.activation_lower_bound, 1.)

        # musculotendon geometry
        musculotendon_len, musculotendon_vel, moment_arm = self.get_musculoskeletal_geometry(joint_state)
        # this is when musculotendon length is too short to accomodate for tendon length
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

        trq_inputs = tf.reduce_sum(- f * moment_arm, axis=-1)

        # first two elements of state are joint position, last two elements are joint angular velocities
        old_pos = tf.cast(joint_state[:, :2], dtype=tf.float32)
        old_vel = tf.cast(joint_state[:, 2:], dtype=tf.float32)
        c2 = tf.cos(old_pos[:, 1])
        s2 = tf.sin(old_pos[:, 1])

        # inertia matrix (batch_size x 2 x 2)
        inertia = self.inertia_c + c2[:, tf.newaxis, tf.newaxis] * self.inertia_m

        # coriolis torques (batch_size x 2) plus a damping term (scaled by self.c_viscosity)
        coriolis_1 = self.coriolis_1 * s2 * (2 * old_vel[:, 0] * old_vel[:, 1] + old_vel[:, 1] ** 2) + \
                     self.c_viscosity * old_vel[:, 0]
        coriolis_2 = self.coriolis_2 * s2 * (old_vel[:, 0] ** 2) + self.c_viscosity * old_vel[:, 1]
        coriolis = tf.stack([coriolis_1, coriolis_2], axis=1)

        rhs = -coriolis[:, :, tf.newaxis] + trq_inputs[:, :, tf.newaxis]

        denom = 1 / (inertia[:, 0, 0] * inertia[:, 1, 1] - inertia[:, 0, 1] * inertia[:, 1, 0])
        l_col = tf.stack([inertia[:, 1, 1], -inertia[:, 1, 0]], axis=1)
        r_col = tf.stack([-inertia[:, 0, 1], inertia[:, 0, 0]], axis=1)
        inertia_inv = denom[:, tf.newaxis, tf.newaxis] * tf.stack([l_col, r_col], axis=2)
        new_acc = tf.squeeze(tf.matmul(inertia_inv, rhs))

        # apply Euler
        new_vel = old_vel + new_acc * self.dt
        new_pos = old_pos + old_vel * self.dt

        # clips to make sure things don't get totally crazy
        new_vel = tf.clip_by_value(new_vel, self.vel_lower_bounds, self.vel_upper_bounds)
        new_pos = tf.clip_by_value(new_pos, self.pos_lower_bounds, self.pos_upper_bounds)

        new_state = tf.concat([new_pos, new_vel], axis=1)
        new_cart_state = self.joint2cartesian(new_state)
        new_muscle_state = tf.concat(
            [
                new_activation,
                muscle_len * self.l0_ce,
                muscle_vel * self.vmax,
            ], axis=1)
        new_geometry_state = tf.concat(
            [
                musculotendon_len,
                musculotendon_vel * self.vmax,
                moment_arm
            ], axis=1)
        return new_state, new_cart_state, new_muscle_state, new_geometry_state

    def get_musculoskeletal_geometry(self, joint_state):
        old_pos, old_vel = tf.split(joint_state[:, :, tf.newaxis], 2, axis=1)
        old_pos = old_pos - np.array([np.pi / 2, 0.]).reshape((1, 2, 1))
        old_pos2 = tf.pow(old_pos, 2)
        musculotendon_len = tf.reduce_sum(old_pos * self.a1, axis=1, keepdims=True) + \
            tf.reduce_sum(old_pos2 * self.a2, axis=1, keepdims=True) + self.a0
        moment_arm = old_pos * self.a2 * 2 + self.a1
        musculotendon_vel = tf.reduce_sum(old_vel * moment_arm, axis=1, keepdims=True)
        return musculotendon_len, musculotendon_vel, moment_arm

    def joint2cartesian(self, joint_pos):
        # compute hand position from joint position
        # reshape to have all time steps lined up in 1st dimension
        joint_pos = tf.reshape(joint_pos, (-1, self.state_dim))
        joint_angle_sum = joint_pos[:, 0] + joint_pos[:, 1]

        c1 = tf.cos(joint_pos[:, 0])
        s1 = tf.sin(joint_pos[:, 0])
        c12 = tf.cos(joint_angle_sum)
        s12 = tf.sin(joint_angle_sum)

        end_pos_x = self.L1 * c1 + self.L2 * c12
        end_pos_y = self.L1 * s1 + self.L2 * s12
        end_vel_x = - (self.L1 * s1 + self.L2 * s12) * joint_pos[:, 2]
        end_vel_y = (self.L1 * c1 + self.L2 * c12) * joint_pos[:, 3]

        end_pos = tf.stack([end_pos_x, end_pos_y, end_vel_x, end_vel_y], axis=1)
        return end_pos

    @staticmethod
    def state2target(state, n_timesteps=1):
        # convert a state array to a target array
        targ = state[:, :, tf.newaxis]
        targ = tf.repeat(targ, n_timesteps, axis=-1)
        targ = tf.transpose(targ, [0, 2, 1])
        return targ

    def draw_random_uniform_states(self, batch_size=1):
        # create a batch of new targets in the correct format for the tensorflow compiler
        sz = (batch_size, self.space_dim)
        lo = self.pos_lower_bounds
        hi = self.pos_upper_bounds
        pos = tf.random.uniform(sz, lo, hi)
        vel = tf.zeros(sz)
        return tf.concat([pos, vel], axis=1)

    def set_state_limit_bounds(self, lb, ub):
        lb = np.array(lb).reshape((-1, 1))  # ensure this is a 2D array
        ub = np.array(ub).reshape((-1, 1))
        bounds = np.hstack((lb, ub))
        bounds = bounds * np.ones((self.dof, 2))  # if one bound pair inputed, broadcast to dof rows
        return bounds

    def get_initial_state(self, batch_size=1, initial_joint_state=None):
        if initial_joint_state is None:
            initial_joint_state = self.draw_random_uniform_states(batch_size=batch_size)
        else:
            batch_size = tf.shape(initial_joint_state)[0]
        initial_cartesian_state = self.joint2cartesian(initial_joint_state)
        activation = tf.ones((batch_size, 1, self.n_muscles)) * self.activation_lower_bound
        musculotendon_len, musculotendon_vel, moment_arm = self.get_musculoskeletal_geometry(initial_joint_state)
        muscle_len = tf.maximum(musculotendon_len - self.l0_se, 0.001 * self.l0_ce)
        initial_muscle_state = tf.concat((activation, muscle_len, musculotendon_vel), axis=1)
        initial_geometry_state = tf.concat((musculotendon_len, musculotendon_vel, moment_arm), axis=1)
        return [initial_joint_state, initial_cartesian_state, initial_muscle_state, initial_geometry_state]


class Skeleton:
    # Base class
    def __init__(self, timestep=0.01, space_dim=2, **kwargs):
        self.dof = kwargs.get('dof', space_dim)  # degrees of freedom of the skeleton (eg number of joints)
        self.space_dim = space_dim  # the dimensionality of the space (eg 2 for cartesian xy space)
        self.input_dim = kwargs.get('input_dim', self.dof)  # dim of the control input (eg torques), usually >= dof
        self.state_dim = kwargs.get('state_dim', self.dof * 2)  # usually position and velocity so twice the dof
        self.output_dim = kwargs.get('output_dim', self.state_dim)  # usually the new state so same as state_dim
        self.geometry_state_dim = 2 + self.dof  # two geometry variable per muscle: path_length, path_velocity
        self.dt = timestep
        self.init = False
        self.built = False

        # handle position & velocity ranges
        pos_lower_bound = kwargs.get('pos_lower_bound', -1.)
        pos_upper_bound = kwargs.get('pos_upper_bound', +1.)
        vel_lower_bound = kwargs.get('vel_lower_bound', -np.inf)
        vel_upper_bound = kwargs.get('vel_upper_bound', +np.inf)
        pos_bounds = self.set_state_limit_bounds(lb=pos_lower_bound, ub=pos_upper_bound)
        vel_bounds = self.set_state_limit_bounds(lb=vel_lower_bound, ub=vel_upper_bound)

        # initialized with the 'initialize_limit_ranges' method below
        self.pos_upper_bounds = pos_bounds[:, 1]
        self.pos_lower_bounds = pos_bounds[:, 0]
        self.vel_upper_bounds = vel_bounds[:, 1]
        self.vel_lower_bounds = vel_bounds[:, 0]

        # default is no delay
        proprioceptive_delay = kwargs.get('proprioceptive_delay', timestep)
        visual_delay = kwargs.get('visual_delay', timestep)
        self.proprioceptive_delay = int(proprioceptive_delay / self.dt)
        self.visual_delay = int(visual_delay / self.dt)

    def setattr(self, name: str, value):
        self.__setattr__(name, value)

    def build(self, timestep=0.01):
        self.dt = timestep
        self.built = True

    @staticmethod
    def state2target(state, n_timesteps=1):
        # convert a state array to a target array
        targ = state[:, :, tf.newaxis]
        targ = tf.repeat(targ, n_timesteps, axis=-1)
        targ = tf.transpose(targ, [0, 2, 1])
        return targ

    def draw_random_uniform_states(self, batch_size=1):
        # create a batch of new targets in the correct format for the tensorflow compiler
        sz = (batch_size, self.dof)
        lo = self.pos_lower_bounds
        hi = self.pos_upper_bounds
        pos = tf.random.uniform(sz, lo, hi)
        vel = tf.zeros(sz)
        return tf.concat([pos, vel], axis=1)

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
        assert np.all(pos > self.pos_lower_bounds)
        assert np.all(pos < self.pos_upper_bounds)
        assert np.all(vel > self.vel_lower_bounds)
        assert np.all(vel < self.vel_upper_bounds)

        pos = tf.cast(pos, dtype=tf.float32)
        vel = tf.cast(vel, dtype=tf.float32)
        states = tf.concat([pos, vel], axis=1)
        tiled_states = tf.tile(states, [batch_size, 1])[:batch_size, :]  # if more than one different positions input
        return tiled_states

    def set_state_limit_bounds(self, lb, ub):
        lb = np.array(lb).reshape((-1, 1))  # ensure this is a 2D array
        ub = np.array(ub).reshape((-1, 1))
        bounds = np.hstack((lb, ub))
        bounds = bounds * np.ones((self.dof, 2))  # if one bound pair inputed, broadcast to dof rows
        return bounds

    def get_geometry(self, path_coordinates, path_fixation_body, muscle_transitions, row_splits, skeleton_state):
        xy, dxy_dt, dxy_ddof = self.path2cartesian(path_coordinates, path_fixation_body, skeleton_state)
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
        segment_len_cleaned = tf.where(muscle_transitions, 0., segment_len)
        segment_vel_cleaned = tf.where(muscle_transitions, 0., segment_vel)
        segment_mom_cleaned = tf.where(muscle_transitions, 0., segment_moments)

        # flip all dimensions to allow making ragged tensors below (you need to do it from the rows)
        segment_len_flipped = tf.transpose(segment_len_cleaned, [2, 1, 0])
        segment_vel_flipped = tf.transpose(segment_vel_cleaned, [2, 1, 0])
        segment_mom_flipped = tf.transpose(segment_mom_cleaned, [2, 1, 0])

        # create ragged tensors, which allows to hold each individual muscle's fixation points on the second dimension
        # in case there is not the same number of fixation point for each muscles
        segment_len_ragged = tf.RaggedTensor.from_row_splits(segment_len_flipped, row_splits=row_splits)
        segment_vel_ragged = tf.RaggedTensor.from_row_splits(segment_vel_flipped, row_splits=row_splits)
        segment_mom_ragged = tf.RaggedTensor.from_row_splits(segment_mom_flipped, row_splits=row_splits)

        # now we can sum all segments' contribution along the ragged dimension, that is along all fixation points for
        # each muscle
        musculotendon_len = tf.reduce_sum(segment_len_ragged, axis=1)
        musculotendon_vel = tf.reduce_sum(segment_vel_ragged, axis=1)
        moments = tf.reduce_sum(segment_mom_ragged, axis=1)

        # pack all this into one state array and flip the dimensions back (batch_size * n_features * n_muscles)
        geometry_state = tf.transpose(tf.concat([musculotendon_len, musculotendon_vel, moments], axis=1), [2, 1, 0])
        return geometry_state

    @staticmethod
    def path2cartesian(path_coordinates, path_fixation_body, skeleton_state):
        return None, None, None

    def clip_velocity(self, pos, vel):
        vel = tf.clip_by_value(vel, self.vel_lower_bounds, self.vel_upper_bounds)
        vel = tf.where(condition=tf.logical_and(vel < 0, pos <= self.pos_lower_bounds), x=tf.zeros_like(vel), y=vel)
        vel = tf.where(condition=tf.logical_and(vel > 0, pos >= self.pos_upper_bounds), x=tf.zeros_like(vel), y=vel)
        return vel


class TwoDofArm(Skeleton):

    def __init__(self, timestep=0.01, **kwargs):
        # TODO reset sho_limits to previous values
        sho_limit = np.deg2rad([-0, 140])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 160])
        lb = (sho_limit[0], elb_limit[0])
        ub = (sho_limit[1], elb_limit[1])
        super().__init__(timestep=timestep, space_dim=2, dof=2, pos_lower_bound=lb, pos_upper_bound=ub, **kwargs)

        self.m1 = 1.864572  # masses of arm links
        self.m2 = 1.534315
        self.L1g = 0.180496  # center of mass of the links
        self.L2g = 0.181479
        self.I1 = 0.013193  # moments of inertia around the center of mass
        self.I2 = 0.020062
        self.L1 = 0.309  # length of links
        self.L2 = 0.26

        # pre-compute values for mass, coriolis, and gravity matrices
        inertia_11_c = self.m1 * self.L1g ** 2 + self.I1 + self.m2 * (self.L2g ** 2 + self.L1 ** 2) + self.I2
        inertia_12_c = self.m2 * (self.L2g ** 2) + self.I2
        inertia_22_c = self.m2 * (self.L2g ** 2) + self.I2
        inertia_11_m = 2 * self.m2 * self.L1 * self.L2g
        inertia_12_m = self.m2 * self.L1 * self.L2g

        inertia_c = np.array([[[inertia_11_c, inertia_12_c],
                               [inertia_12_c, inertia_22_c]]]).astype(np.float32)
        inertia_m = np.array([[[inertia_11_m, inertia_12_m],
                               [inertia_12_m, 0.]]]).astype(np.float32)
        self.inertia_c = inertia_c.reshape((1, 2, 2))  # 0-th axis is for broadcasting to batch_size when used
        self.inertia_m = inertia_m.reshape((1, 2, 2))

        self.coriolis_1 = -self.m2 * self.L1 * self.L2g
        self.coriolis_2 = self.m2 * self.L1 * self.L2g
        self.c_viscosity = 0.0  # put at zero but available if implemented later on

    def __call__(self, inputs, skeleton_state, joint_load=np.zeros(1)):
        # joint_load is the set of torques applied at the endpoint of the two-link arm (i.e. applied at the 'hand')
        # first two elements of state are joint position, last two elements are joint angular velocities
        old_vel = tf.cast(skeleton_state[:, 2:], dtype=tf.float32)
        old_pos = tf.cast(skeleton_state[:, :2], dtype=tf.float32)
        c1 = tf.cos(old_pos[:, 0])
        c2 = tf.cos(old_pos[:, 1])
        c12 = tf.cos(old_pos[:, 0] + old_pos[:, 1])
        s1 = tf.sin(old_pos[:, 0])
        s2 = tf.sin(old_pos[:, 1])
        s12 = tf.sin(old_pos[:, 0] + old_pos[:, 1])

        # inertia matrix (batch_size x 2 x 2)
        inertia = self.inertia_c + c2[:, tf.newaxis, tf.newaxis] * self.inertia_m

        # coriolis torques (batch_size x 2) plus a damping term (scaled by self.c_viscosity)
        coriolis_1 = self.coriolis_1 * s2 * (2 * old_vel[:, 0] * old_vel[:, 1] + old_vel[:, 1] ** 2) + \
            self.c_viscosity * old_vel[:, 0]
        coriolis_2 = self.coriolis_2 * s2 * (old_vel[:, 0] ** 2) + self.c_viscosity * old_vel[:, 1]
        coriolis = tf.stack([coriolis_1, coriolis_2], axis=1)

        # jacobian to distribute external loads (torques) applied at endpoint to the two rigid links
        jacobian_11 = c1 * self.L1 + c12 * self.L2
        jacobian_12 = c12 * self.L2
        jacobian_21 = s1 * self.L1 + s12 * self.L2
        jacobian_22 = s12 * self.L2

        # apply external loads
        loads = tf.constant(joint_load, shape=(self.input_dim,), dtype=tf.float32)
        r_col = (jacobian_11 * loads[0]) + (jacobian_21 * loads[1])  # these are torques
        l_col = (jacobian_12 * loads[0]) + (jacobian_22 * loads[1])
        torques = inputs + tf.stack([r_col, l_col], axis=1)

        rhs = -coriolis[:, :, tf.newaxis] + torques[:, :, tf.newaxis]

        denom = 1 / (inertia[:, 0, 0] * inertia[:, 1, 1] - inertia[:, 0, 1] * inertia[:, 1, 0])
        l_col = tf.stack([inertia[:, 1, 1], -inertia[:, 1, 0]], axis=1)
        r_col = tf.stack([-inertia[:, 0, 1], inertia[:, 0, 0]], axis=1)
        inertia_inv = denom[:, tf.newaxis, tf.newaxis] * tf.stack([l_col, r_col], axis=2)
        new_acc = tf.squeeze(tf.matmul(inertia_inv, rhs))

        new_vel = old_vel + new_acc * self.dt  # Euler
        new_pos = old_pos + old_vel * self.dt

        # clips to make sure things don't get totally crazy
        new_vel = self.clip_velocity(new_pos, new_vel)
        new_pos = tf.clip_by_value(new_pos, self.pos_lower_bounds, self.pos_upper_bounds)

        new_state = tf.concat([new_pos, new_vel], axis=1)
        return new_state

    def joint2cartesian(self, joint_pos):
        # compute hand position from joint position
        # reshape to have all time steps lined up in 1st dimension
        joint_pos = tf.reshape(joint_pos, (-1, self.state_dim))
        joint_angle_sum = joint_pos[:, 0] + joint_pos[:, 1]

        c1 = tf.cos(joint_pos[:, 0])
        s1 = tf.sin(joint_pos[:, 0])
        c12 = tf.cos(joint_angle_sum)
        s12 = tf.sin(joint_angle_sum)

        end_pos_x = self.L1 * c1 + self.L2 * c12
        end_pos_y = self.L1 * s1 + self.L2 * s12
        end_vel_x = - (self.L1 * s1 + self.L2 * s12) * joint_pos[:, 2]
        end_vel_y = (self.L1 * c1 + self.L2 * c12) * joint_pos[:, 3]

        end_pos = tf.stack([end_pos_x, end_pos_y, end_vel_x, end_vel_y], axis=1)
        return end_pos

    def path2cartesian(self, path_coordinates, path_fixation_body, skeleton_state):
        n_points = path_fixation_body.size
        joint_angles, joint_vel = tf.split(skeleton_state, 2, axis=-1)
        sho, elb_wrt_sho = tf.split(joint_angles, 2, axis=-1)
        elb = elb_wrt_sho + sho
        elb_y = self.L1 * tf.sin(sho)[:, :, tf.newaxis]
        elb_x = self.L1 * tf.cos(sho)[:, :, tf.newaxis]

        # If we want the position of a fixation point relative to the origin of its bone given global cartesian
        # coordinates, then we use the joint angles; here we are trying to do the inverse of that, that is getting the
        # fixation point in global cartesian coordinates given its position relative to the origin of the bone it is
        # fixed on. Therefore we use a minus in front of the angles since we are doing the inverse rotation.
        # This line picks no rotation angle if the muscle path point is fixed on the extrinstic workspace
        # (path_fixation_body = 0), the shoulder angle if it is fixed on the upper arm (path_fixation_body = 1) and the
        # eblow angle if it is fixed on the forearm (path_fixation_body = 2).
        ang = tf.where(path_fixation_body.flatten() == 0., 0., tf.where(path_fixation_body.flatten() == 1., -sho, -elb))
        ca = tf.cos(ang)
        sa = tf.sin(ang)

        # rotation matrix to transform the bone-relative coordinates into global coordinates
        rot1 = tf.reshape(tf.concat([ca, sa], axis=1), (-1, 2, n_points))
        rot2 = tf.reshape(tf.concat([-sa, ca], axis=1), (-1, 2, n_points))

        # derivative of each fixation point's position wrt the angle of the bone they are fixed on
        dx_da = tf.reduce_sum(-path_coordinates * rot2, axis=1, keepdims=True)
        dy_da = tf.reduce_sum(path_coordinates * rot1, axis=1, keepdims=True)

        # Derivative of each fixation point's position wrt each angle
        # This is counter-intuitive but the derivative of any point wrt the shoulder angle (da1) is equal to the
        # derivative of that point wrt the angle of the bone they are actually fixed on (dx_da or dy_da), even if that
        # bone is the forearm and not the upper arm. However, if the bone is indeed the forearm, then an additional term
        # must be added (see below).
        dx_da1 = tf.where(path_fixation_body == 0., 0., dx_da) + tf.where(path_fixation_body == 2., -elb_y, 0.)
        dy_da1 = tf.where(path_fixation_body == 0., 0., dy_da) + tf.where(path_fixation_body == 2., elb_x, 0.)
        dx_da2 = tf.where(path_fixation_body == 2., dx_da, 0.)
        dy_da2 = tf.where(path_fixation_body == 2., dy_da, 0.)

        dxy_da1 = tf.concat([dx_da1, dy_da1], axis=1)
        dxy_da2 = tf.concat([dx_da2, dy_da2], axis=1)
        dxy_da = tf.concat([dxy_da1[:, :, tf.newaxis, :], dxy_da2[:, :, tf.newaxis, :]], axis=2)

        sho_vel_3d = joint_vel[:, 0, tf.newaxis, tf.newaxis]
        elb_vel_3d = joint_vel[:, 1, tf.newaxis, tf.newaxis] + sho_vel_3d
        dxy_dt = dxy_da1 * sho_vel_3d + dxy_da2 * elb_vel_3d  # by virtue of the chain rule

        bone_origin = tf.where(path_fixation_body == 2, tf.concat([elb_x, elb_y], axis=1), 0.)
        xy = tf.concat([dy_da, -dx_da], axis=1) + bone_origin
        return xy, dxy_dt, dxy_da


class PointMass(Skeleton):
    def __init__(self, mass=1., **kwargs):
        super().__init__(**kwargs)
        self.mass = mass

    def __call__(self, inputs, skeleton_state, joint_load=np.zeros(1)):
        joint_load = tf.constant(joint_load, shape=(1, self.dof), dtype=tf.float32)
        new_acc = inputs + joint_load  # load will broadcast to match batch_size

        old_vel = tf.cast(skeleton_state[:, self.dof:], dtype=tf.float32)
        old_pos = tf.cast(skeleton_state[:, :self.dof], dtype=tf.float32)
        new_vel = old_vel + new_acc * self.dt / self.mass  # Euler
        new_pos = old_pos + old_vel * self.dt

        new_vel = self.clip_velocity(new_pos, new_vel)
        new_pos = tf.clip_by_value(new_pos, self.pos_lower_bounds, self.pos_upper_bounds)
        new_state = tf.concat([new_pos, new_vel], axis=1)
        return new_state

    def path2cartesian(self, path_coordinates, path_fixation_body, skeleton_state):
        pos, vel = tf.split(skeleton_state[:, :, tf.newaxis], 2, axis=1)
        # if fixed on the point mass, then add the point-mass position / velocity to the fixation point coordinate
        pos = tf.where(path_fixation_body == 0, 0., pos) + path_coordinates
        vel = tf.where(path_fixation_body == 0, 0., vel)
        dpos_ddof = tf.one_hot(tf.range(0, self.dof), self.dof)[tf.newaxis, :, :, tf.newaxis]
        dpos_ddof = tf.where(path_fixation_body == 0, 0., dpos_ddof)
        return pos, vel, dpos_ddof

    @staticmethod
    def joint2cartesian(joint_pos):
        return joint_pos

