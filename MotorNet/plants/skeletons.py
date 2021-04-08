import numpy as np
import tensorflow as tf


class Arm:
    def __init__(self, timestep=0.01, **kwargs):
        self.dof = 2  # degrees of freedom of the skeleton (eg number of joints)
        self.space_dim = 2  # the dimensionality of the space (eg 2 for cartesian xy space)
        self.input_dim = 6  # 6 muscles
        self.state_dim = self.dof * 2  # usually position and velocity so twice the dof
        self.output_dim = self.state_dim  # usually the new state so same as state_dim
        self.muscle_state_dim = 5
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

        self.m1 = 2.10  # masses of arm links
        self.m2 = 1.65
        self.L1g = 0.146  # center of mass of the links
        self.L2g = 0.179
        self.I1 = 0.024  # moments of inertia around the center of mass
        self.I2 = 0.025
        self.L1 = 0.335  # length of links
        self.L2 = 0.263

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

    def __call__(self, excitation, joint_state, muscle_state):

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
        muscle_len, muscle_vel, moment_arm = self.get_musculoskeletal_geometry(joint_state)

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
                moment_arm
            ], axis=1)
        return new_state, new_cart_state, new_muscle_state

    def get_musculoskeletal_geometry(self, joint_state):
        old_pos, old_vel = tf.split(joint_state[:, :, tf.newaxis], 2, axis=1)
        old_pos = old_pos - np.array([np.pi / 2, 0.]).reshape((1, 2, 1))
        old_pos2 = tf.pow(old_pos, 2)
        muscle_len = (tf.reduce_sum(old_pos * self.a1, axis=1, keepdims=True) +
                      tf.reduce_sum(old_pos2 * self.a2, axis=1, keepdims=True) +
                      self.a0 - self.l0_se) / self.l0_ce
        # this is when musculotendon length is too short to accomodate for tendon length
        muscle_len = tf.maximum(muscle_len, 0.001)
        moment_arm = old_pos * self.a2 * 2 + self.a1
        # muscle velocity is already normalized by vmax in the fvce equation of the __call__ method
        muscle_vel = tf.reduce_sum(old_vel * moment_arm, axis=1, keepdims=True)
        return muscle_len, muscle_vel, moment_arm

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

    def get_initial_states(self, batch_size=1, initial_joint_state=None):
        if initial_joint_state is None:
            initial_joint_state = self.draw_random_uniform_states(batch_size=batch_size)
        else:
            batch_size = tf.shape(initial_joint_state)[0]
        initial_cartesian_state = self.joint2cartesian(initial_joint_state)
        activation = tf.ones((batch_size, 1, self.n_muscles)) * self.activation_lower_bound
        muscle_len, muscle_vel, moment_arm = self.get_musculoskeletal_geometry(initial_joint_state)
        initial_muscle_state = tf.concat((activation, muscle_len, muscle_vel, moment_arm), axis=1)
        return [initial_joint_state, initial_cartesian_state, initial_muscle_state]
