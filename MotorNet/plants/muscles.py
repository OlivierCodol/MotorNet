import numpy as np
import tensorflow as tf


class Muscle:
    # --------------------------
    # base class for muscles
    # --------------------------
    def __init__(self, input_dim=1, state_dim=1, output_dim=1, min_activation=0., tau_activation=0.015, tau_deactivation=0.05):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.state_name = []
        self.output_dim = output_dim
        self.min_activation = min_activation
        self.tau_activation = tau_activation
        self.tau_deactivation = tau_deactivation
        self.to_build_dict = {'timestep': [],
                              'max_isometric_force': []}
        self.dt = None
        self.n_muscles = None
        self.max_iso_force = None
        self.vmax = None
        self.l0_se = None
        self.l0_ce = None
        self.l0_pe = None
        self.built = False

    def setattr(self, name: str, value):
        self.__setattr__(name, value)

    @staticmethod
    def get_initial_muscle_state(batch_size, geometry):
        return None

    def build(self, timestep, max_isometric_force, **kwargs):
        self.dt = timestep
        self.n_muscles = np.array(max_isometric_force).size
        self.max_iso_force = tf.reshape(tf.cast(max_isometric_force, dtype=tf.float32), (1, 1, self.n_muscles))
        self.vmax = tf.ones((1, 1, self.n_muscles), dtype=tf.float32)
        self.l0_se = tf.ones((1, 1, self.n_muscles), dtype=tf.float32)
        self.l0_ce = tf.ones((1, 1, self.n_muscles), dtype=tf.float32)
        self.l0_pe = tf.ones((1, 1, self.n_muscles), dtype=tf.float32)
        self.built = True

    def activation_ode(self, excitation, muscle_state):
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1])
        excitation = tf.reshape(excitation, (-1, 1, self.n_muscles))
        activation = tf.clip_by_value(activation, self.min_activation, 1.)
        excitation = tf.clip_by_value(excitation, self.min_activation, 1.)

        tau_scaler = 0.5 + 1.5 * activation
        tau = tf.where(excitation > activation, self.tau_activation * tau_scaler, self.tau_deactivation / tau_scaler)
        d_activation = (excitation - activation) / tau
        new_activation = activation + d_activation * self.dt
        new_activation = tf.clip_by_value(new_activation, self.min_activation, 1.)
        return new_activation


class ReluMuscle(Muscle):
    # --------------------------
    # A rectified linear muscle that outputs the input directly, but can only have a positive activation value.
    # --------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_name = ['excitation',
                           'force']
        self.state_dim = len(self.state_name)

    def __call__(self, excitation, *args, **kwargs):
        excitation = excitation[:, tf.newaxis, :]
        forces = tf.nn.relu(excitation) * self.max_iso_force
        muscle_state = tf.concat([excitation, forces], axis=1)
        return forces, muscle_state

    def get_initial_muscle_state(self, batch_size, geometry):
        excitation0 = tf.ones((batch_size, 1, self.n_muscles)) * self.min_activation
        force0 = tf.zeros((batch_size, 1, self.n_muscles))
        muscle_state0 = tf.concat([excitation0, force0], axis=1)
        return muscle_state0


class RigidTendonHillMuscle(Muscle):
    # --------------------------
    # This is based on Kistemaker et al 2006
    # --------------------------

    def __init__(self, min_activation=0.001, **kwargs):

        super().__init__(min_activation=min_activation, **kwargs)

        self.state_name = ['activation',
                           'muscle length',
                           'muscle_velocity',
                           'force-length PE',
                           'force-length CE',
                           'force-velocity CE']
        self.state_dim = len(self.state_name)

        # parameters for the passive element (PE) and contractile element (CE)
        self.pe_k = 5.
        self.pe_1 = self.pe_k / 0.66
        self.pe_den = tf.exp(self.pe_k) - 1
        self.ce_gamma = 0.45
        self.ce_Af = 0.25
        self.ce_fmlen = 1.4

        # pre-define attributes:
        self.musculotendon_slack_len = None
        self.max_iso_force = None
        self.l0_se = None
        self.l0_ce = None
        self.l0_pe = None
        self.vmax = None
        self.k_pe = None
        self.s_as = 0.001
        self.f_iso_n_den = .66 ** 2
        self.b_rel_st_den = 5e-3 - 0.3
        self.k_se = 1 / (0.04 ** 2)
        self.q_crit = 0.3

        self.to_build_dict = {'timestep': [],
                              'max_isometric_force': [],
                              'tendon_length': [],
                              'optimal_muscle_length': []}
        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        tendon_length = kwargs.get('tendon_length')
        optimal_muscle_length = kwargs.get('optimal_muscle_length')
        self.dt = timestep
        self.n_muscles = np.array(tendon_length).size
        self.max_iso_force = tf.reshape(tf.cast(max_isometric_force, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_se = tf.reshape(tf.cast(tendon_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_ce = tf.reshape(tf.cast(optimal_muscle_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_pe = self.l0_ce * 1.15
        self.musculotendon_slack_len = self.l0_pe + self.l0_se
        self.vmax = 10 * self.l0_ce
        self.k_pe = 1 / ((1.66 - self.l0_pe / self.l0_ce) ** 2)
        self.built = True

    def get_initial_muscle_state(self, batch_size, geometry):
        musculotendon_len = tf.slice(geometry, [0, 0, 0], [-1, 1, -1])
        muscle_len = tf.maximum(musculotendon_len - self.l0_se, 0.)
        activation = tf.ones_like(muscle_len) * self.min_activation
        return tf.concat([activation, muscle_len, tf.zeros((batch_size, 4, self.n_muscles))], axis=1)

    def __call__(self, excitation, muscle_state, geometry_state):
        new_activation = self.activation_ode(excitation, muscle_state)

        # musculotendon geometry
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_vel = tf.slice(geometry_state, [0, 1, 0], [-1, 1, -1])
        muscle_len = tf.maximum(musculotendon_len - self.l0_se, 0.)
        muscle_strain = tf.maximum((muscle_len - self.l0_pe) / self.l0_ce, 0.)
        muscle_len_n = muscle_len / self.l0_ce
        muscle_vel_n = muscle_vel / self.vmax

        # muscle forces
        flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 3.)
        flce = tf.maximum(1 + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den, 0.01)

        a_rel_st = tf.where(muscle_len_n > 1., .41 * flce, .41)
        b_rel_st = tf.where(
            condition=new_activation < self.q_crit,
            x=5.2 * (1 - .9 * ((new_activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
            y=5.2)
        dvdf_isom_con = b_rel_st / (new_activation * (flce + a_rel_st))  # slope at isometric point wrt concentric curve
        dfdvcon0 = 1. / dvdf_isom_con

        p1 = -(flce * new_activation * .5) / (self.s_as - dfdvcon0 * 2.)
        p2 = ((flce * new_activation * .5) ** 2) / (self.s_as - dfdvcon0 * 2.)
        p3 = -1.5 * flce * new_activation
        p4 = -self.s_as

        nom = tf.where(
            condition=muscle_vel_n < 0,
            x=muscle_vel_n * new_activation * a_rel_st + flce * new_activation * b_rel_st,
            y=-p1 * p3 - p1 * p4 * muscle_vel_n + p2 - p3 * muscle_vel_n - p4 * muscle_vel_n ** 2)
        den = tf.where(condition=muscle_vel_n < 0, x=b_rel_st - muscle_vel_n, y=p1 + muscle_vel_n)
        active_force = tf.maximum(nom / den, 0.)

        force = (active_force + flpe) * self.max_iso_force
        new_muscle_state = tf.concat([new_activation, muscle_len, muscle_vel, flpe, flce, active_force], axis=1)
        return force, new_muscle_state


class RigidTendonHillMuscleThelen(Muscle):
    # --------------------------
    # This is based on Thelen et al 2003
    # --------------------------
    def __init__(self, min_activation=0.001, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)

        self.state_name = ['activation',
                           'muscle length',
                           'muscle_velocity',
                           'force-length PE',
                           'force-length CE',
                           'force-velocity CE']
        self.state_dim = len(self.state_name)

        # parameters for the passive element (PE) and contractile element (CE)
        self.pe_k = 5.
        self.pe_1 = self.pe_k / 0.66
        self.pe_den = tf.exp(self.pe_k) - 1
        self.ce_gamma = 0.45
        self.ce_Af = 0.25
        self.ce_fmlen = 1.4

        # pre-define attributes:
        self.musculotendon_slack_len = None
        self.max_iso_force = None
        self.l0_se = None
        self.l0_ce = None
        self.l0_pe = None
        self.vmax = None
        self.ce_0 = None
        self.ce_1 = None
        self.ce_2 = None
        self.ce_3 = None
        self.ce_4 = None
        self.ce_5 = None
        self.ce_6 = None
        self.ce_7 = None
        self.n_muscles = None

        self.to_build_dict = {'timestep': [],
                              'max_isometric_force': [],
                              'tendon_length': [],
                              'optimal_muscle_length': []}
        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        tendon_length = kwargs.get('tendon_length')
        optimal_muscle_length = kwargs.get('optimal_muscle_length')

        self.n_muscles = np.array(tendon_length).size
        self.dt = timestep
        self.max_iso_force = tf.reshape(tf.cast(max_isometric_force, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_se = tf.reshape(tf.cast(tendon_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_ce = tf.reshape(tf.cast(optimal_muscle_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_pe = self.l0_ce * 1.
        self.musculotendon_slack_len = self.l0_pe + self.l0_se

        # pre-computed for speed
        self.vmax = 10 * self.l0_ce
        self.ce_0 = 3. * self.vmax
        self.ce_1 = 3. * self.ce_Af * self.vmax * self.ce_fmlen
        self.ce_2 = 3. * self.ce_Af * self.vmax
        self.ce_3 = 8. * self.ce_Af * self.ce_fmlen
        self.ce_4 = self.ce_Af * self.ce_fmlen
        self.ce_5 = self.ce_Af * self.vmax
        self.ce_6 = 8. * self.ce_fmlen
        self.ce_7 = self.ce_Af * self.ce_fmlen * self.vmax

        self.built = True

    def get_initial_muscle_state(self, batch_size, geometry):
        musculotendon_len = tf.slice(geometry, [0, 0, 0], [-1, 1, -1])
        muscle_len = tf.maximum(musculotendon_len - self.l0_se, 0.)
        activation = tf.ones_like(muscle_len) * self.min_activation
        return tf.concat([activation, muscle_len, tf.zeros((batch_size, 4, self.n_muscles))], axis=1)

    def __call__(self, excitation, muscle_state, geometry_state):
        new_activation = self.activation_ode(excitation, muscle_state)

        # musculotendon geometry
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_len = tf.maximum(musculotendon_len - self.l0_se, 0.001)
        muscle_vel = tf.slice(geometry_state, [0, 1, 0], [-1, 1, -1])

        # muscle forces
        new_activation3 = new_activation * 3.
        nom = tf.where(condition=muscle_vel <= 0,
                       x=self.ce_Af * (new_activation * self.ce_0 + 4. * muscle_vel + self.vmax),
                       y=(self.ce_1 * new_activation
                          - self.ce_2 * new_activation
                          + self.ce_3 * muscle_vel
                          + self.ce_7
                          - self.ce_5
                          + self.ce_6 * muscle_vel))
        den = tf.where(condition=muscle_vel <= 0,
                       x=(new_activation3 * self.ce_5 + self.ce_5 - 4. * muscle_vel),
                       y=(new_activation3 * self.ce_7
                          - new_activation3 * self.ce_5
                          + self.ce_7
                          + 8. * self.ce_Af * muscle_vel
                          - self.ce_5
                          + 8. * muscle_vel))
        force_vel_ce = tf.maximum(nom / den, 0.)
        force_len_pe = tf.maximum((tf.exp(self.pe_1 * (muscle_len - self.l0_pe) / self.l0_ce) - 1) / self.pe_den, 0.)
        force_len_ce = tf.exp((- ((muscle_len / self.l0_ce) - 1) ** 2) / self.ce_gamma)

        force = (new_activation * force_len_ce * force_vel_ce + force_len_pe) * self.max_iso_force
        new_muscle_state = tf.concat(
            [new_activation, muscle_len, muscle_vel, force_len_pe, force_len_ce, force_vel_ce], axis=1)
        return force, new_muscle_state
