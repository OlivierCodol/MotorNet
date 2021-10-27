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

        self.pos_lower_bound = kwargs.get('pos_lower_bound', -1.)
        self.pos_upper_bound = kwargs.get('pos_upper_bound', +1.)
        self.vel_lower_bound = kwargs.get('vel_lower_bound', -np.inf)
        self.vel_upper_bound = kwargs.get('vel_upper_bound', +np.inf)

        self._call_fn = Lambda(lambda x: self.call(*x))
        self._path2cartesian_fn = Lambda(lambda x: self._path2cartesian(*x))
        self._joint2cartesian_fn = Lambda(lambda x: self._joint2cartesian(x))
        self._clip_velocity_fn = Lambda(lambda x: self._clip_velocity(*x))
        self.init = False
        self.built = False

    def build(self, timestep, pos_upper_bound, pos_lower_bound, vel_upper_bound, vel_lower_bound):
        self.pos_upper_bound = pos_upper_bound
        self.pos_lower_bound = pos_lower_bound
        self.vel_upper_bound = vel_upper_bound
        self.vel_lower_bound = vel_lower_bound
        self.dt = timestep
        self.built = True

    def __call__(self, inputs, joint_state, **kwargs):
        """This is a wrapper method to prevent memory leaks"""
        endpoint_load = kwargs.get('endpoint_load', self.default_endpoint_load)
        return self._call_fn((inputs, joint_state, endpoint_load))

    def setattr(self, name: str, value):
        self.__setattr__(name, value)

    @abstractmethod
    def call(self, *args):
        return

    def joint2cartesian(self, joint_state):
        return self._joint2cartesian_fn(joint_state)

    def path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
        return self._path2cartesian_fn((path_coordinates, path_fixation_body, joint_state))

    def clip_velocity(self, pos, vel):
        return self._clip_velocity_fn((pos, vel))

    @abstractmethod
    def _path2cartesian(self, *args, **kwargs):
        return

    def _clip_velocity(self, pos, vel):
        vel = tf.clip_by_value(vel, self.vel_lower_bound, self.vel_upper_bound)
        vel = tf.where(condition=tf.logical_and(vel < 0, pos <= self.pos_lower_bound), x=tf.zeros_like(vel), y=vel)
        vel = tf.where(condition=tf.logical_and(vel > 0, pos >= self.pos_upper_bound), x=tf.zeros_like(vel), y=vel)
        return vel

    @abstractmethod
    def _joint2cartesian(self, *args, **kwargs):
        return

    def get_base_config(self):
        cfg = {'dof': self.dof, 'dt': self.dt, 'space_dim': self.space_dim}
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

        self.m1 = kwargs.get('m1', 1.864572)  # masses of arm links
        self.m2 = kwargs.get('m2', 1.534315)
        self.L1g = kwargs.get('L1g', 0.180496)  # center of mass of the links
        self.L2g = kwargs.get('L2g', 0.181479)
        self.I1 = kwargs.get('I1', 0.013193)  # moments of inertia around the center of mass
        self.I2 = kwargs.get('I2', 0.020062)
        self.L1 = kwargs.get('L1', 0.309)  # length of links
        self.L2 = kwargs.get('L2', 0.26)

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

    def call(self, inputs, joint_state, endpoint_load):
        # first two elements of state are joint position, last two elements are joint angular velocities
        old_vel = tf.cast(joint_state[:, 2:], dtype=tf.float32)
        old_pos = tf.cast(joint_state[:, :2], dtype=tf.float32)
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
        # loads = tf.cast(endpoint_load, dtype=tf.float32)
        r_col = (jacobian_11 * endpoint_load[:, 0]) + (jacobian_21 * endpoint_load[:, 1])  # these are torques
        l_col = (jacobian_12 * endpoint_load[:, 0]) + (jacobian_22 * endpoint_load[:, 1])
        torques = inputs + tf.stack([r_col, l_col], axis=1)

        rhs = -coriolis[:, :, tf.newaxis] + torques[:, :, tf.newaxis]

        denom = 1 / (inertia[:, 0, 0] * inertia[:, 1, 1] - inertia[:, 0, 1] * inertia[:, 1, 0])
        l_col = tf.stack([inertia[:, 1, 1], -inertia[:, 1, 0]], axis=1)
        r_col = tf.stack([-inertia[:, 0, 1], inertia[:, 0, 0]], axis=1)
        inertia_inv = denom[:, tf.newaxis, tf.newaxis] * tf.stack([l_col, r_col], axis=2)
        new_acc_3d = tf.matmul(inertia_inv, rhs)
        new_acc = new_acc_3d[:, :, 0]  # somehow tf.squeeze doesn't work well in a Lambda wrap...

        new_vel = old_vel + new_acc * self.dt  # Euler
        new_pos = old_pos + old_vel * self.dt

        # clips to make sure things don't get totally crazy
        new_vel = self.clip_velocity(new_pos, new_vel)
        new_pos = tf.clip_by_value(new_pos, self.pos_lower_bound, self.pos_upper_bound)

        new_state = tf.concat([new_pos, new_vel], axis=1)
        return new_state

    def _joint2cartesian(self, joint_state):
        # compute cartesian state from joint state
        # reshape to have all time steps lined up in 1st dimension
        joint_state = tf.reshape(joint_state, (-1, self.state_dim))
        joint_angle_sum = joint_state[:, 0] + joint_state[:, 1]

        c1 = tf.cos(joint_state[:, 0])
        s1 = tf.sin(joint_state[:, 0])
        c12 = tf.cos(joint_angle_sum)
        s12 = tf.sin(joint_angle_sum)

        end_pos_x = self.L1 * c1 + self.L2 * c12
        end_pos_y = self.L1 * s1 + self.L2 * s12
        end_vel_x = - (self.L1 * s1 + self.L2 * s12) * joint_state[:, 2]
        end_vel_y = (self.L1 * c1 + self.L2 * c12) * joint_state[:, 3]

        end_pos = tf.stack([end_pos_x, end_pos_y, end_vel_x, end_vel_y], axis=1)
        return end_pos

    def _path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
        n_points = tf.size(path_fixation_body).numpy()
        joint_angles, joint_vel = tf.split(joint_state, 2, axis=-1)
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
        flat_path_fixation_body = tf.reshape(path_fixation_body, -1)
        ang = tf.where(flat_path_fixation_body == 0., 0., tf.where(flat_path_fixation_body == 1., -sho, -elb))
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

    def get_save_config(self):
        cfg = self.get_base_config()
        cfg.update({'I1': self.I1, 'I2': self.I2, 'L1': self.L1, 'L2': self.L2, 'L1g': self.L1g, 'L2g': self.L2g,
                    'c_viscosity': self.c_viscosity, 'coriolis_1': self.coriolis_1, 'coriolis_2': self.coriolis_2,
                    'm1': self.m1, 'm2': self.m2})
        return cfg


class PointMass(Skeleton):

    def __init__(self, space_dim, mass=1., **kwargs):
        super().__init__(dof=space_dim, space_dim=space_dim, **kwargs)
        self.mass = mass

    def call(self, inputs, joint_state, endpoint_load):
        new_acc = inputs + endpoint_load  # load will broadcast to match batch_size

        old_vel = tf.cast(joint_state[:, self.dof:], dtype=tf.float32)
        old_pos = tf.cast(joint_state[:, :self.dof], dtype=tf.float32)
        new_vel = old_vel + new_acc * self.dt / self.mass  # Euler
        new_pos = old_pos + old_vel * self.dt

        new_vel = self.clip_velocity(new_pos, new_vel)
        new_pos = tf.clip_by_value(new_pos, self.pos_lower_bound, self.pos_upper_bound)
        new_state = tf.concat([new_pos, new_vel], axis=1)
        return new_state

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
