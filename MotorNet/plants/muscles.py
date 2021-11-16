import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from abc import abstractmethod
# todo add all the _get format to self.lambdas

class Muscle:
    """
    Base class for muscles.
    """
    def __init__(self, input_dim=1, output_dim=1, min_activation=0., tau_activation=0.015, tau_deactivation=0.05):
        self.input_dim = input_dim
        self.state_name = []
        self.output_dim = output_dim
        self.min_activation = tf.constant(min_activation, name='min_activation')
        self.tau_activation = tf.constant(tau_activation, name='tau_activation')
        self.tau_deactivation = tf.constant(tau_deactivation, name='tau_deactivation')
        self.to_build_dict = {'max_isometric_force': []}
        self.to_build_dict_default = {}
        self.dt = None
        self.n_muscles = None
        self.max_iso_force = None
        self.vmax = None
        self.l0_se = None
        self.l0_ce = None
        self.l0_pe = None
        self._get_initial_muscle_state_fn = Lambda(function=lambda x: self._get_initial_muscle_state(*x),
                                                   name='get_initial_muscle_state')
        self._activation_ode_fn = Lambda(lambda x: self._activation_ode(*x), name='activation_ode')
        self._slice_states = Lambda(lambda x: tf.slice(x[0], [0, x[1], 0], [-1, x[2], -1]), name='muscle_slice_states')
        self._concat = Lambda(lambda x: tf.concat(x[0], axis=x[1]), name='muscle_concat')
        self._zeros_like = Lambda(lambda x: tf.zeros_like(x), name='lambda_zeros_like')
        self._ones_like = Lambda(lambda x: tf.ones_like(x), name='lambda_ones_like')
        self._zeros = Lambda(lambda x: tf.zeros(x), name='lambda_zeros')
        self._ones = Lambda(lambda x: tf.ones(x), name='lambda_ones')
        self._maximum = Lambda(lambda x: tf.maximum(x[0], x[1]), name='lambda_maximum')
        self._minimum = Lambda(lambda x: tf.minimum(x[0], x[1]), name='lambda_minimum')
        self.clip_activation = Lambda(lambda a: tf.clip_by_value(a, self.min_activation, 1.), name='clip_activation')
        self._get_tau = Lambda(lambda x: tf.where(
            condition=x[0] > x[1],
            x=self.tau_activation * (0.5 + 1.5 * x[1]),
            y=self.tau_deactivation / (0.5 + 1.5 * x[1])))
        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        self.dt = tf.constant(timestep, name='dt')
        self.n_muscles = np.array(max_isometric_force).size
        self.vmax = tf.constant(tf.ones((1, 1, self.n_muscles), dtype=tf.float32), name='vmax')
        self.l0_se = tf.constant(tf.ones((1, 1, self.n_muscles), dtype=tf.float32), name='l0_se')
        self.l0_ce = tf.constant(tf.ones((1, 1, self.n_muscles), dtype=tf.float32), name='l0_ce')
        self.l0_pe = tf.constant(tf.ones((1, 1, self.n_muscles), dtype=tf.float32), name='l0_pe')
        self.max_iso_force = tf.reshape(
            tf.constant(max_isometric_force, dtype=tf.float32, name='max_iso_force'), (1, 1, self.n_muscles))
        self.built = True

    def activation_ode(self, excitation, muscle_state):
        """This wrap allows inspecting the argument names when the function is called."""
        return self._activation_ode_fn((excitation, muscle_state))

    def get_initial_muscle_state(self, batch_size, geometry_state):
        return self._get_initial_muscle_state_fn((batch_size, geometry_state))

    @abstractmethod
    def _get_initial_muscle_state(self, batch_size, geometry_state):
        return

    @abstractmethod
    def integrate(self, dt, state_derivative, muscle_state, geometry_state):
        return

    def update_ode(self, excitation, muscle_state):
        activation = self._slice_states((muscle_state, 0, 1))
        return self.activation_ode(excitation, activation)

    def _activation_ode(self, excitation, activation):
        excitation = self.clip_activation(tf.reshape(excitation, (-1, 1, self.n_muscles)))
        activation = self.clip_activation(activation)
        return (excitation - activation) / self._get_tau((excitation, activation))

    def setattr(self, name: str, value):
        self.__setattr__(name, value)

    def get_save_config(self):
        cfg = {'name': str(self.__name__)}
        return cfg


class ReluMuscle(Muscle):
    # --------------------------
    # A rectified linear muscle that outputs the input directly, but can only have a positive activation value.
    # --------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = 'ReluMuscle'

        self.state_name = ['excitation',
                           'muscle lenth',
                           'muscle velocity',
                           'force']
        self.state_dim = len(self.state_name)

    def integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = self._slice_states((muscle_state, 0, 1)) + state_derivative * dt
        activation = self.clip_activation(activation)
        forces = self._maximum((activation, 0.)) * self.max_iso_force
        muscle_len = self._slice_states((geometry_state, 0, 1))
        muscle_vel = self._slice_states((geometry_state, 1, 1))
        return self._concat(([activation, muscle_len, muscle_vel, forces], 1))

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        excitation0 = self._ones((batch_size, 1, self.n_muscles)) * self.min_activation
        force0 = self._zeros((batch_size, 1, self.n_muscles))
        len_vel = self._slice_states((geometry_state, 0, 2))
        return self._concat(([excitation0, len_vel, force0], 1))


class RigidTendonHillMuscle(Muscle):
    """
    This Hill-type muscle implementation is based on Kistemaker et al. (2006)
    """

    def __init__(self, min_activation=0.001, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = 'RigidTendonHillMuscle'

        self.state_name = ['activation',
                           'muscle length',
                           'muscle velocity',
                           'force-length PE',
                           'force-length CE',
                           'force-velocity CE',
                           'force']
        self.state_dim = len(self.state_name)

        # parameters for the passive element (PE) and contractile element (CE)
        self.pe_k = tf.constant(5., name='pe_k')
        self.pe_1 = tf.constant(self.pe_k / 0.66, name='pe_1')
        self.pe_den = tf.constant(tf.exp(self.pe_k) - 1, name='pe_den')
        self.ce_gamma = tf.constant(0.45, name='ce_gamma')
        self.ce_Af = tf.constant(0.25, name='ce_Af')
        self.ce_fmlen = tf.constant(1.4, name='ce_fmlen')

        # pre-define attributes:
        self.musculotendon_slack_len = None
        self.k_pe = None
        self.s_as = tf.constant(0.001, name='s_as')
        self.f_iso_n_den = tf.constant(.66 ** 2, name='f_iso_n_den')
        self.k_se = tf.constant(1 / (0.04 ** 2), name='k_se')
        self.q_crit = tf.constant(0.3, name='q_crit')
        self.b_rel_st_den = tf.constant(5e-3 - self.q_crit, name='b_rel_st_den')
        self.min_flce = tf.constant(0.01, name='min_flce')

        self.to_build_dict = {'max_isometric_force': [],
                              'tendon_length': [],
                              'optimal_muscle_length': [],
                              'normalized_slack_muscle_length': []}
        self.to_build_dict_default = {'normalized_slack_muscle_length': 1.4}

        self._get_p1 = Lambda(lambda x: -(x[0] * .5) / (self.s_as - x[1] * 2.), name='get_p1')
        self._get_p2 = Lambda(lambda x: ((x[0] * .5) ** 2) / (self.s_as - x[1] * 2.), name='get_p2')
        self._get_p3 = Lambda(lambda f_x_a: -1.5 * f_x_a, name='get_p3')

        self._get_flce = Lambda(
            function=lambda muscle_len_n: 1 + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den,
            name='get_flce')

        self._get_a_rel_st = Lambda(
            function=lambda x: tf.where(x[0] > 1., .41 * x[1], .41),
            name='get_a_rel_st')

        self._get_b_rel_st = Lambda(
            function=lambda activation: tf.where(
                condition=activation < self.q_crit,
                x=5.2 * (1 - .9 * ((activation - self.q_crit) / self.b_rel_st_den)) ** 2,
                y=5.2),
            name='get_b_rel_st')

        self._get_active_force_nom = Lambda(
            function=lambda x: tf.where(
                condition=x[0] < 0,
                x=x[0] * x[1] * x[2] + x[3] * x[4],
                y=-x[5] * x[6] + x[5] * self.s_as * x[0] + x[7] - x[6] * x[0] + self.s_as * x[0] ** 2),
            name='get_active_force_nom')

        self._get_active_force_den = Lambda(
            function=lambda x: tf.where(condition=x[0] < 0, x=x[1] - x[0], y=x[2] + x[0]),
            name='get_active_force_den')

        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        tendon_length = kwargs.get('tendon_length')
        optimal_muscle_length = kwargs.get('optimal_muscle_length')
        normalized_slack_muscle_length = kwargs.get('normalized_slack_muscle_length')

        self.dt = tf.constant(timestep, name='dt')
        self.n_muscles = np.array(tendon_length).size
        self.l0_ce = tf.reshape(
            tensor=tf.constant(optimal_muscle_length, dtype=tf.float32, name='l0_ce'),
            shape=(1, 1, self.n_muscles))
        self.l0_pe = tf.constant(self.l0_ce * normalized_slack_muscle_length, name='l0_pe')
        self.k_pe = tf.constant(1 / ((1.66 - self.l0_pe / self.l0_ce) ** 2), name='k_pe')
        self.l0_se = tf.reshape(tf.constant(tendon_length, dtype=tf.float32, name='l0_se'), (1, 1, self.n_muscles))
        self.max_iso_force = tf.reshape(
            tensor=tf.constant(max_isometric_force, dtype=tf.float32, name='max_iso_force'),
            shape=(1, 1, self.n_muscles))
        self.musculotendon_slack_len = tf.constant(self.l0_pe + self.l0_se, name='musculotendon_slack_len')
        self.vmax = tf.constant(10 * self.l0_ce, name='vmax')
        self.built = True

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = self._slice_states((geometry_state, 0, 1))
        muscle_state = self._ones_like(musculotendon_len) * self.min_activation
        return self.integrate(self.dt, self._zeros_like(musculotendon_len), muscle_state, geometry_state)

    def integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = self.clip_activation(self._slice_states((muscle_state, 0, 1)) + state_derivative * dt)

        # musculotendon geometry
        musculotendon_len = self._slice_states((geometry_state, 0, 1))
        muscle_vel = self._slice_states((geometry_state, 1, 1))
        muscle_len = self._maximum((musculotendon_len - self.l0_se, 0.))
        muscle_strain = self._maximum(((muscle_len - self.l0_pe) / self.l0_ce, 0.))
        muscle_len_n = muscle_len / self.l0_ce
        muscle_vel_n = muscle_vel / self.vmax

        # muscle forces
        flpe = self._minimum((self.k_pe * (muscle_strain ** 2), 3.))
        flce = self._maximum((self._get_flce(muscle_len_n), self.min_flce))

        a_rel_st = self._get_a_rel_st((muscle_len_n, flce))
        b_rel_st = self._get_b_rel_st(activation)
        dfdvcon0 = activation * (flce + a_rel_st) / b_rel_st  # inv of slope at isometric point wrt concentric curve

        f_x_a = flce * activation  # to speed up computation

        p1 = self._get_p1((f_x_a, dfdvcon0))
        p2 = self._get_p2((f_x_a, dfdvcon0))
        p3 = self._get_p3(f_x_a)

        nom = self._get_active_force_nom((muscle_vel_n, activation, a_rel_st, f_x_a, b_rel_st, p1, p3, p2))
        den = self._get_active_force_den((muscle_vel_n, b_rel_st, p1))
        active_force = self._maximum((nom / den, 0.))

        force = (active_force + flpe) * self.max_iso_force
        return self._concat(([activation, muscle_len, muscle_vel, flpe, flce, active_force, force], 1))


class RigidTendonHillMuscleThelen(Muscle):
    # --------------------------
    # This is based on Thelen et al 2003
    # --------------------------
    def __init__(self, min_activation=0.001, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = 'RigidTendonHillMuscleThelen'

        self.state_name = ['activation',
                           'muscle length',
                           'muscle velocity',
                           'force-length PE',
                           'force-length CE',
                           'force-velocity CE',
                           'force']
        self.state_dim = len(self.state_name)

        # parameters for the passive element (PE) and contractile element (CE)
        self.pe_k = tf.constant(5., name='pe_k')
        self.pe_1 = tf.constant(self.pe_k / 0.66, name='pe_1')
        self.pe_den = tf.constant(tf.exp(self.pe_k) - 1, name='pe_den')
        self.ce_gamma = tf.constant(0.45, name='ce_gamma')
        self.ce_Af = tf.constant(0.25, name='ce_Af')
        self.ce_fmlen = tf.constant(1.4, name='ce_fmlen')

        # pre-define attributes:
        self.musculotendon_slack_len = None
        self.ce_0 = None
        self.ce_1 = None
        self.ce_2 = None
        self.ce_3 = None
        self.ce_4 = None
        self.ce_5 = None

        self.to_build_dict = {'max_isometric_force': [],
                              'tendon_length': [],
                              'optimal_muscle_length': [],
                              'normalized_slack_muscle_length': []}
        self.to_build_dict_default = {'normalized_slack_muscle_length': 1.}

        self._get_flce = Lambda(
            lambda muscle_len: tf.exp((- ((muscle_len / self.l0_ce) - 1) ** 2) / self.ce_gamma),
            name='get_flce')

        self._get_flpe_tmp = Lambda(
            lambda muscle_len:
            (tf.exp(self.pe_1 * (muscle_len - self.l0_pe) / self.l0_ce) - 1) / self.pe_den,
            name='get_flpe_tmp')

        self._get_muscle_vel_nom = Lambda(
            lambda x: tf.where(
                condition=x[1] <= 0,
                x=self.ce_Af * (x[0] * self.ce_0 + 4. * x[1] + self.vmax),
                y=self.ce_2 * x[0] + self.ce_3 * x[1] + self.ce_4),
            name='get_muscle_vel_nom')

        self._get_muscle_vel_den = Lambda(
            lambda x: tf.where(
                condition=x[1] <= 0,
                x=x[0] * self.ce_1 + self.ce_1 - 4. * x[1],
                y=self.ce_4 * x[0] + self.ce_5 * x[1] + self.ce_4),
            name='get_muscle_vel_den')

        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        tendon_length = kwargs.get('tendon_length')
        optimal_muscle_length = kwargs.get('optimal_muscle_length')
        normalized_slack_muscle_length = kwargs.get('normalized_slack_muscle_length')

        self.n_muscles = np.array(tendon_length).size
        self.dt = tf.constant(timestep, name='dt')

        self.max_iso_force = tf.reshape(
            tensor=tf.constant(max_isometric_force, dtype=tf.float32, name='max_iso_force'),
            shape=(1, 1, self.n_muscles))
        self.l0_ce = tf.reshape(
            tensor=tf.constant(optimal_muscle_length, dtype=tf.float32, name='l0_ce'),
            shape=(1, 1, self.n_muscles))
        self.l0_pe = tf.constant(self.l0_ce * normalized_slack_muscle_length, name='l0_pe')
        self.l0_se = tf.reshape(tf.constant(tendon_length, dtype=tf.float32, name='l0_se'), (1, 1, self.n_muscles))
        self.musculotendon_slack_len = tf.constant(self.l0_pe + self.l0_se, name='musculotendon_slack_len')
        self.vmax = tf.constant(10 * self.l0_ce, name='vmax')

        # pre-computed for speed
        self.ce_0 = tf.constant(3. * self.vmax, name='ce_0')
        self.ce_1 = tf.constant(self.ce_Af * self.vmax, name='ce_1')
        self.ce_2 = tf.constant(3. * self.ce_Af * self.vmax * self.ce_fmlen - 3. * self.ce_Af * self.vmax, name='ce_2')
        self.ce_3 = tf.constant(8. * self.ce_Af * self.ce_fmlen + 8. * self.ce_fmlen, name='ce_3')
        self.ce_4 = tf.constant(self.ce_Af * self.ce_fmlen * self.vmax - self.ce_1, name='ce_4')
        self.ce_5 = tf.constant(8. * (self.ce_Af + 1.), name='ce_5')

        self.built = True

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = self._slice_states((geometry_state, 0, 1))
        muscle_state = self._ones_like(musculotendon_len) * self.min_activation
        return self.integrate(self.dt, self._zeros_like(musculotendon_len), muscle_state, geometry_state)

    def integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = self._slice_states((muscle_state, 0, 1)) + state_derivative * dt
        activation = self.clip_activation(activation)

        # musculotendon geometry
        musculotendon_len = self._slice_states((geometry_state, 0, 1))
        muscle_len = self._maximum((musculotendon_len - self.l0_se, 0.001))
        muscle_vel = self._slice_states((geometry_state, 1, 1))

        # muscle forces
        nom = self._get_muscle_vel_nom((activation, muscle_vel))
        den = self._get_muscle_vel_den((activation * 3, muscle_vel))
        fvce = self._maximum((nom / den, 0.))
        flpe_tmp = self._get_flpe_tmp(muscle_len)
        flpe = self._maximum((flpe_tmp, 0.))
        flce = self._get_flce(muscle_len)
        force = (activation * flce * fvce + flpe) * self.max_iso_force
        return self._concat(([activation, muscle_len, muscle_vel, flpe, flce, fvce, force], 1))


class CompliantTendonHillMuscle(RigidTendonHillMuscle):

    def __init__(self, min_activation=0.01, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = 'CompliantTendonHillMuscle'

        self.state_name = [
            'activation',
            'muscle length',
            'muscle velocity',
            'force-length PE',
            'force-length SE',
            'active force',
            'force']
        self.state_dim = len(self.state_name)

        # Lambda layers
        self._get_muscle_ode_sqrt_term = Lambda(
            function=lambda x:
                x[0] ** 2 + 2 * x[0] * x[1] * self.s_as +
                2 * x[0] * x[3] + x[1] ** 2 * (self.s_as ** 2) +
                2 * x[1] * x[3] * self.s_as + x[2] + x[3] ** 2,
            name='get_muscle_ode_sqrt_term')

        self._get_muscle_ode_cond = Lambda(
            function=lambda x: tf.where(tf.logical_and(x[0] < 0, x[1] >= x[2]), -1, 1),
            name='get_muscle_ode_cond')

        self._get_muscle_vel_nom = Lambda(
            function=lambda x: tf.where(
                condition=x[0] < x[1], x=x[2] * (x[0] - x[1]), y=-x[0] + x[3] * self.s_as - x[4] - tf.sqrt(x[5])),
            name='get_muscle_vel_nom')

        self._get_muscle_vel_den = Lambda(
            function=lambda x: tf.where(condition=x[0] < x[1], x=x[0] + x[2] * x[3], y=-2 * self.s_as),
            name='get_muscle_vel_den')

        self.f_iso_n_tmp = Lambda(
            function=lambda muscle_len_n: 1 + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den,
            name='f_iso_n_tmp')

        # # defensive code to ensure this p2 never explode to inf (this way p2 is divided before it is multiplied)
        self._get_p2_containing_term = Lambda(
            function=lambda x: (4 * ((x[0] * 0.5) ** 2) * (- self.s_as)) / (self.s_as - x[1] * 2),
            name='get_p2_containing_term')

        # if musculotendon length is negative, raise an error.
        # if musculotendon length is less than tendon slack length, assign all (most of) the length to the tendon.
        # if musculotendon length is more than tendon slack length and less than musculotendon slack length, assign to
        #   the tendon up to the tendon slack length, and the rest to the muscle length.
        # if musculotendon length is more than tendon slack length and muscle slack length combined, find the muscle
        #   length that satisfies equilibrium between tendon passive forces and muscle passive forces.
        self._get_initial_muscle_len = Lambda(
            function=lambda musculotendon_len: tf.where(
                condition=musculotendon_len < 0,
                x=-1,
                y=tf.where(
                    condition=musculotendon_len < self.l0_se,
                    x=0.001 * self.l0_ce,
                    y=tf.where(
                        condition=musculotendon_len < self.l0_se + self.l0_pe,
                        x=musculotendon_len - self.l0_se,
                        y=(self.k_pe * self.l0_pe * self.l0_se ** 2 -
                            self.k_se * (self.l0_ce ** 2) * musculotendon_len +
                            self.k_se * self.l0_ce ** 2 * self.l0_se -
                            self.l0_ce * self.l0_se * tf.sqrt(self.k_pe * self.k_se)
                            * (-musculotendon_len + self.l0_pe + self.l0_se)) /
                          (self.k_pe * self.l0_se ** 2 - self.k_se * self.l0_ce ** 2)))))

        self.built = False

    def integrate(self, dt, state_derivative, muscle_state, geometry_state):
        """Perform the numerical integration given the current states and their derivatives"""

        # Compute musculotendon geometry
        muscle_len = self._slice_states((muscle_state, 1, 1))
        muscle_len_n = muscle_len / self.l0_ce
        musculotendon_len = self._slice_states((geometry_state, 0, 1))
        tendon_len = musculotendon_len - muscle_len
        tendon_strain = self._maximum(((tendon_len - self.l0_se) / self.l0_se, 0.))
        muscle_strain = self._maximum(((muscle_len - self.l0_pe) / self.l0_ce, 0.))

        # Compute forces
        flse = tf.minimum(self.k_se * (tendon_strain ** 2), 1.)
        flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 1.)
        active_force = self._maximum((flse - flpe, 0.))

        # Integrate
        d_activation = self._slice_states((state_derivative, 0, 1))
        muscle_vel_n = self._slice_states((state_derivative, 1, 1))
        activation = self._slice_states((muscle_state, 0, 1)) + d_activation * dt
        activation = self.clip_activation(activation)
        new_muscle_len = (muscle_len_n + dt * muscle_vel_n) * self.l0_ce

        muscle_vel = muscle_vel_n * self.vmax
        force = flse * self.max_iso_force
        return self._concat(([activation, new_muscle_len, muscle_vel, flpe, flse, active_force, force], 1))

    def update_ode(self, excitation, muscle_state):
        activation = self._slice_states((muscle_state, 0, 1))
        d_activation = self.activation_ode(excitation, activation)
        muscle_len_n = self._slice_states((muscle_state, 1, 1)) / self.l0_ce
        active_force = self._slice_states((muscle_state, 5, 1))
        new_muscle_vel_n = self.muscle_ode(muscle_len_n, activation, active_force)
        return self._concat(([d_activation, new_muscle_vel_n], 1))

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = self._slice_states((geometry_state, 0, 1))
        # musculotendon_vel = tf.slice(geometry_state, [0, 1, 0], [-1, 1, -1])
        activation = self._ones_like(musculotendon_len) * self.min_activation

        muscle_len = self._get_initial_muscle_len(musculotendon_len)
        # tf.debugging.assert_non_negative(muscle_len, message='initial muscle length was < 0.')
        tendon_len = musculotendon_len - muscle_len
        tendon_strain = self._maximum(((tendon_len - self.l0_se) / self.l0_se, 0.))
        muscle_strain = self._maximum(((muscle_len - self.l0_pe) / self.l0_ce, 0.))

        # Compute forces
        flse = self._minimum((self.k_se * (tendon_strain ** 2), 1.))
        flpe = self._minimum((self.k_pe * (muscle_strain ** 2), 1.))
        active_force = self._maximum((flse - flpe, 0.))

        muscle_vel_n = self.muscle_ode(muscle_len / self.l0_ce, activation, active_force)
        muscle_state = self._concat(([activation, muscle_len], 1))
        state_derivative = self._concat(([self._zeros_like(musculotendon_len), muscle_vel_n], 1))

        return self.integrate(self.dt, state_derivative, muscle_state, geometry_state)

    def muscle_ode(self, muscle_len_n, activation, active_force):
        f_iso_n_tmp = self.f_iso_n_tmp(muscle_len_n)
        f_iso_n = self._maximum((f_iso_n_tmp, self.min_flce))
        a_rel_st = self._get_a_rel_st((muscle_len_n, f_iso_n))
        b_rel_st = self._get_b_rel_st(activation)
        # inv of slope at isometric point wrt concentric curve
        f_x_a = f_iso_n * activation  # to speed up computation
        dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

        p1 = self._get_p1((f_x_a, dfdvcon0))
        p3 = self._get_p3(f_x_a)
        p2_containing_term = self._get_p2_containing_term((f_x_a, dfdvcon0))

        # defensive code to avoid propagation of negative square root in the non-selected tf.where outcome
        # the assertion is to ensure that any selected item is indeed not a negative root.
        sqrt_term = self._get_muscle_ode_sqrt_term((active_force, p1, p2_containing_term, p3))
        cond = self._get_muscle_ode_cond((sqrt_term, active_force, f_x_a))
        tf.debugging.assert_non_negative(cond, message='root that should be used is negative.')
        sqrt_term = self._maximum((sqrt_term, 0.))

        muscle_vel_nom = self._get_muscle_vel_nom((active_force, f_x_a, b_rel_st, p1, p3, sqrt_term))
        muscle_vel_den = self._get_muscle_vel_den((active_force, f_x_a, activation, a_rel_st))
        return muscle_vel_nom / muscle_vel_den
