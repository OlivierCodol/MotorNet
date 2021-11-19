import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from abc import abstractmethod


class Skeleton:
    """
    Base class for skeletons.
    """

    def __init__(self, dof, space_dim, **kwargs):
        self.dof = dof  # degrees of freedom of the skeleton (eg number of joints)
        self.space_dim = space_dim  # the dimensionality of the space (eg 2 for cartesian xy space)
        self.input_dim = kwargs.get('input_dim', self.dof)  # dim of the control input (eg torques), usually >= dof
        self.state_dim = kwargs.get('state_dim', self.dof * 2)  # usually position and velocity so twice the dof
        self.output_dim = kwargs.get('output_dim', self.state_dim)  # usually the new state so same as state_dim
        self.geometry_state_dim = 2 + self.dof  # two geometry variable per muscle: path_length, path_velocity
        self.default_endpoint_load = tf.zeros((1, self.space_dim), dtype=tf.float32)
        self.dt = None
        self.half_dt = None
        self.integration_method = None

        self.pos_lower_bound = kwargs.get('pos_lower_bound', -1.)
        self.pos_upper_bound = kwargs.get('pos_upper_bound', +1.)
        self.vel_lower_bound = kwargs.get('vel_lower_bound', -1000.)  # cap as defensive code
        self.vel_upper_bound = kwargs.get('vel_upper_bound', +1000.)

        self._update_ode_fn = Lambda(lambda x: self._update_ode(*x), name='skeleton_update_ode')
        self._clip_velocity_fn = Lambda(lambda x: self._clip_velocity(*x), name='skeleton_clip_velocity')
        self._integrate_fn = Lambda(lambda x: self._integrate(*x), name='skeleton_integrate')
        self._path2cartesian_fn = Lambda(lambda x: self._path2cartesian(*x), name='path2cartesian')
        self._joint2cartesian_fn = Lambda(lambda x: self._joint2cartesian(*x), name='joint2cartesian')
        self.clip_position = None
        self._call_fn = None
        self.init = False
        self.built = False

    def build(self, timestep, pos_upper_bound, pos_lower_bound, vel_upper_bound, vel_lower_bound,
              integration_method: str = 'euler'):
        self.pos_upper_bound = tf.constant(pos_upper_bound, name='pos_upper_bound')
        self.pos_lower_bound = tf.constant(pos_lower_bound, name='pos_lower_bound')
        self.vel_upper_bound = tf.constant(vel_upper_bound, name='vel_upper_bound')
        self.vel_lower_bound = tf.constant(vel_lower_bound, name='vel_lower_bound')
        self.dt = tf.constant(timestep, name='dt')
        self.half_dt = tf.constant(self.dt / 2, name='half_dt')
        self.integration_method = integration_method.casefold()  # make string fully in lower case
        self.clip_position = Lambda(
            function=lambda x: tf.clip_by_value(x, self.pos_lower_bound, self.pos_upper_bound), name='clip_position')

        if self.integration_method == 'euler':
            self._call_fn = Lambda(lambda x: self._euler(*x), name='skeleton_euler_integration')
        elif self.integration_method in ('rk4', 'rungekutta4', 'runge-kutta4', 'runge-kutta-4'):  # tuple faster thn set
            self._call_fn = Lambda(lambda x: self._rungekutta4(*x), name='skeleton_rk4_integration')
        else:
            raise ValueError(' ''integration_method'' should be ''euler'' or ''rk4''.')
        self.built = True

    def __call__(self, inputs, joint_state, **kwargs):
        """This is a wrapper method to prevent memory leaks"""
        endpoint_load = kwargs.get('endpoint_load', self.default_endpoint_load)
        return self._call_fn((inputs, joint_state, endpoint_load))

    def setattr(self, name: str, value):
        self.__setattr__(name, value)

    def _euler(self, inputs, joint_state, endpoint_load):
        state_derivative = self.update_ode(inputs, joint_state, endpoint_load)
        return self.integrate(self.dt, state_derivative=state_derivative, joint_state=joint_state)

    def _rungekutta4(self, inputs, joint_state, endpoint_load):
        k1 = self.update_ode(inputs, joint_state, endpoint_load)
        state = self.integrate(self.half_dt, state_derivative=k1, joint_state=joint_state)
        k2 = self.update_ode(inputs, state, endpoint_load)
        state = self.integrate(self.half_dt, state_derivative=k2, joint_state=state)
        k3 = self.update_ode(inputs, state, endpoint_load)
        state = self.integrate(self.dt, state_derivative=k3, joint_state=state)
        k4 = self.update_ode(inputs, state, endpoint_load)
        k = (k1 + 2 * (k2 + k3) + k4) / 6
        return self.integrate(self.dt, state_derivative=k, joint_state=joint_state)

    def integrate(self, dt, state_derivative, joint_state):
        return self._integrate_fn((dt, state_derivative, joint_state))

    def joint2cartesian(self, *args, **kwargs):
        return self._joint2cartesian_fn((*args, *[v for v in kwargs.values()]))

    def update_ode(self, inputs, joint_state, endpoint_load):
        return self._update_ode_fn((inputs, joint_state, endpoint_load))

    def path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
        return self._path2cartesian_fn((path_coordinates, path_fixation_body, joint_state))

    def clip_velocity(self, pos, vel):
        return self._clip_velocity_fn((pos, vel))

    @abstractmethod
    def _path2cartesian(self, *args, **kwargs):
        return

    @abstractmethod
    def _integrate(self, dt, state_derivative, joint_state):
        return

    @abstractmethod
    def _joint2cartesian(self, *args, **kwargs):
        return

    @abstractmethod
    def _update_ode(self, *args, **kwargs):
        return

    def _clip_velocity(self, pos, vel):
        vel = tf.clip_by_value(vel, self.vel_lower_bound, self.vel_upper_bound)
        vel = tf.where(condition=tf.logical_and(vel < 0, pos <= self.pos_lower_bound), x=0., y=vel)
        vel = tf.where(condition=tf.logical_and(vel > 0, pos >= self.pos_upper_bound), x=0., y=vel)
        return vel

    def get_base_config(self):
        cfg = {'dof': self.dof, 'dt': str(self.dt.numpy()), 'space_dim': self.space_dim}
        return cfg

    def get_save_config(self):
        return self.get_base_config()


class TwoDofArm(Skeleton):

    def __init__(self, **kwargs):
        sho_limit = np.deg2rad([-0, 140])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 160])
        lb = (sho_limit[0], elb_limit[0])
        ub = (sho_limit[1], elb_limit[1])
        super().__init__(dof=2, space_dim=2, pos_lower_bound=lb, pos_upper_bound=ub, **kwargs)

        self.m1 = tf.constant(kwargs.get('m1', 1.864572), name='skeleton_m1')  # masses of arm links
        self.m2 = tf.constant(kwargs.get('m2', 1.534315), name='skeleton_m2')
        self.L1g = tf.constant(kwargs.get('L1g', 0.180496), name='skeleton_L1g')  # center of mass of the links
        self.L2g = tf.constant(kwargs.get('L2g', 0.181479), name='skeleton_L2g')
        self.I1 = tf.constant(kwargs.get('I1', 0.013193), name='skeleton_I1')  # moment of inertia around center of mass
        self.I2 = tf.constant(kwargs.get('I2', 0.020062), name='skeleton_I2')
        self.L1 = tf.constant(kwargs.get('L1', 0.309), name='skeleton_L1')  # length of links
        self.L2 = tf.constant(kwargs.get('L2', 0.26), name='skeleton_L2')

        # pre-compute values for mass, coriolis, and gravity matrices
        inertia_11_c = self.m1 * self.L1g ** 2 + self.I1 + self.m2 * (self.L2g ** 2 + self.L1 ** 2) + self.I2
        inertia_12_c = self.m2 * (self.L2g ** 2) + self.I2
        inertia_22_c = self.m2 * (self.L2g ** 2) + self.I2
        inertia_11_m = 2 * self.m2 * self.L1 * self.L2g
        inertia_12_m = self.m2 * self.L1 * self.L2g
        inertia_c = np.array([[[inertia_11_c, inertia_12_c], [inertia_12_c, inertia_22_c]]]).astype(np.float32)
        inertia_m = np.array([[[inertia_11_m, inertia_12_m], [inertia_12_m, 0.]]]).astype(np.float32)
        self.inertia_c = tf.constant(inertia_c.reshape((1, 2, 2)), name='inertia_c')
        self.inertia_m = tf.constant(inertia_m.reshape((1, 2, 2)), name='inertia_m')

        self.coriolis_1 = tf.constant(- self.m2 * self.L1 * self.L2g, name='coriolis_1')
        self.coriolis_2 = tf.constant(self.m2 * self.L1 * self.L2g, name='coriolis_2')
        self.c_viscosity = tf.constant(0., name='c_viscosity')  # put at zero but available if implemented later on

    def _update_ode(self, inputs, joint_state, endpoint_load):
        # first two elements of state are joint position, last two elements are joint angular velocities
        pos0, pos1, vel0, vel1 = joint_state[:, 0], joint_state[:, 1], joint_state[:, 2], joint_state[:, 3]
        pos_sum = pos0 + pos1
        c1 = tf.cos(pos0)
        c2 = tf.cos(pos1)
        c12 = tf.cos(pos_sum)
        s1 = tf.sin(pos0)
        s2 = tf.sin(pos1)
        s12 = tf.sin(pos_sum)

        # inertia matrix (batch_size x 2 x 2)
        inertia = self.inertia_c + c2[:, tf.newaxis, tf.newaxis] * self.inertia_m

        # coriolis torques (batch_size x 2) plus a damping term (scaled by self.c_viscosity)
        coriolis_1 = (self.coriolis_1 * s2 * (2 * vel0 + vel1)) * vel1 + self.c_viscosity * vel0
        coriolis_2 = (self.coriolis_2 * s2 * vel0) * vel0 + self.c_viscosity * vel1
        coriolis = tf.stack([coriolis_1, coriolis_2], axis=1)

        # jacobian to distribute external loads (torques) applied at endpoint to the two rigid links
        jacobian_11 = c1 * self.L1 + c12 * self.L2
        jacobian_12 = c12 * self.L2
        jacobian_21 = s1 * self.L1 + s12 * self.L2
        jacobian_22 = s12 * self.L2

        # apply external loads
        r_col = (jacobian_11 * endpoint_load[:, 0]) + (jacobian_21 * endpoint_load[:, 1])  # these are torques
        l_col = (jacobian_12 * endpoint_load[:, 0]) + (jacobian_22 * endpoint_load[:, 1])
        torques = inputs + tf.stack([r_col, l_col], axis=1)

        rhs = -coriolis[:, :, tf.newaxis] + torques[:, :, tf.newaxis]

        denom = 1 / (inertia[:, 0, 0] * inertia[:, 1, 1] - inertia[:, 0, 1] * inertia[:, 1, 0])
        l_col = tf.stack([inertia[:, 1, 1], -inertia[:, 1, 0]], axis=1)
        r_col = tf.stack([-inertia[:, 0, 1], inertia[:, 0, 0]], axis=1)
        inertia_inv = denom[:, tf.newaxis, tf.newaxis] * tf.stack([l_col, r_col], axis=2)
        new_acc_3d = tf.matmul(inertia_inv, rhs)
        return new_acc_3d[:, :, 0]  # somehow tf.squeeze doesn't work well in a Lambda wrap...

    def _integrate(self, dt, state_derivative, joint_state):
        old_pos, old_vel = tf.split(joint_state, 2, axis=1)
        new_vel = old_vel + state_derivative * dt
        new_pos = old_pos + old_vel * dt

        # Clip to ensure values don't get off-hand.
        # We clip position after velocity to ensure any off-space position is taken into account when clipping velocity.
        new_vel = self.clip_velocity(new_pos, new_vel)
        new_pos = self.clip_position(new_pos)
        return tf.concat([new_pos, new_vel], axis=1)

    def _joint2cartesian(self, joint_state):
        # compute cartesian state from joint state
        # reshape to have all time steps lined up in 1st dimension
        j = tf.reshape(joint_state, shape=(-1, self.state_dim))
        pos0, pos1, vel0, vel1 = j[:, 0], j[:, 1], j[:, 2], j[:, 3]
        pos_sum = pos0 + pos1

        c1 = tf.cos(pos0)
        s1 = tf.sin(pos0)
        c12 = tf.cos(pos_sum)
        s12 = tf.sin(pos_sum)

        end_pos_x = self.L1 * c1 + self.L2 * c12
        end_pos_y = self.L1 * s1 + self.L2 * s12
        end_vel_x = - (self.L1 * s1 + self.L2 * s12) * vel0
        end_vel_y = (self.L1 * c1 + self.L2 * c12) * vel1
        return tf.stack([end_pos_x, end_pos_y, end_vel_x, end_vel_y], axis=1)

    def _path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
        n_points = tf.size(path_fixation_body)
        joint_angles, joint_vel = tf.split(joint_state, 2, axis=1)

        sho, elb_wrt_sho = tf.split(joint_angles, 2, axis=1)
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
        flat_path_fixation_body = tf.squeeze(tf.reshape(path_fixation_body, shape=(-1, 1)))
        ang = tf.where(flat_path_fixation_body == 0., 0., tf.where(flat_path_fixation_body == 1., -sho, -elb))
        ca = tf.cos(ang)
        sa = tf.sin(ang)

        # rotation matrix to transform the bone-relative coordinates into global coordinates
        rot1 = tf.reshape(tf.concat([ca, sa], axis=1), shape=(-1, 2, n_points))
        rot2 = tf.reshape(tf.concat([-sa, ca], axis=1), shape=(-1, 2, n_points))

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

    def get_save_config(self):
        cfg = self.get_base_config()
        cfg.update({'I1': str(self.I1.numpy()), 'I2': str(self.I2.numpy()), 'L1': str(self.L1.numpy()),
                    'L2': str(self.L2.numpy()), 'L1g': str(self.L1g.numpy()), 'L2g': str(self.L2g.numpy()),
                    'c_viscosity': str(self.c_viscosity.numpy()), 'coriolis_1': str(self.coriolis_1.numpy()),
                    'coriolis_2': str(self.coriolis_2.numpy()), 'm1': str(self.m1.numpy()), 'm2': str(self.m2.numpy())})
        return cfg


class PointMass(Skeleton):

    def __init__(self, space_dim, mass=1., **kwargs):
        super().__init__(dof=space_dim, space_dim=space_dim, **kwargs)
        self.mass = tf.constant(mass, name='mass')

    def _update_ode(self, inputs, joint_state, endpoint_load):
        return inputs + endpoint_load

    def _integrate(self, dt, state_derivative, joint_state):
        old_pos, old_vel = tf.split(joint_state, 2, axis=1)
        new_vel = old_vel + state_derivative * dt / self.mass
        new_pos = old_pos + old_vel * dt
        new_vel = self.clip_velocity(new_pos, new_vel)
        new_pos = self.clip_position(new_pos)
        return tf.concat([new_pos, new_vel], axis=1)

    def _path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
        pos, vel = tf.split(joint_state[:, :, tf.newaxis], 2, axis=1)
        # if fixed on the point mass, then add the point-mass position / velocity to the fixation point coordinate
        pos = tf.where(path_fixation_body == 0, 0., pos) + path_coordinates
        vel = tf.where(path_fixation_body == 0, 0., vel)
        dpos_ddof = tf.one_hot(tf.range(0, self.dof), self.dof)[tf.newaxis, :, :, tf.newaxis]
        dpos_ddof = tf.where(path_fixation_body == 0, 0., dpos_ddof)
        return pos, vel, dpos_ddof

    @staticmethod
    def _joint2cartesian(joint_state):
        return joint_state

    def get_save_config(self):
        cfg = self.get_base_config()
        cfg['mass'] = self.mass
        return cfg
