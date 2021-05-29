import numpy as np
import tensorflow as tf


class Muscle:
    # --------------------------
    # base class for muscles
    # --------------------------
    def __init__(self, input_dim=1, output_dim=1, min_activation=0., tau_activation=0.015, tau_deactivation=0.05):
        self.input_dim = input_dim
        self.state_name = []
        self.output_dim = output_dim
        self.min_activation = min_activation
        self.tau_activation = tau_activation
        self.tau_deactivation = tau_deactivation
        self.to_build_dict = {'timestep': [], 'max_isometric_force': []}
        self.to_build_dict_default = {}
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
    def get_initial_muscle_state(batch_size, geometry_state):
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
        self.__name__ = 'ReluMuscle'

        self.state_name = ['excitation',
                           'muscle lenth',
                           'muscle velocity',
                           'force']
        self.state_dim = len(self.state_name)

    def __call__(self, excitation, muscle_state, geometry_state, **kwargs):
        excitation = excitation[:, tf.newaxis, :]
        forces = tf.nn.relu(excitation) * self.max_iso_force
        muscle_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_vel = tf.slice(geometry_state, [0, 1, 0], [-1, 1, -1])
        muscle_state = tf.concat([excitation, muscle_len, muscle_vel, forces], axis=1)
        return forces, muscle_state

    def get_initial_muscle_state(self, batch_size, geometry_state):
        excitation0 = tf.ones((batch_size, 1, self.n_muscles)) * self.min_activation
        force0 = tf.zeros((batch_size, 1, self.n_muscles))
        len_vel = tf.slice(geometry_state, [0, 0, 0], [-1, 2, -1])
        muscle_state0 = tf.concat([excitation0, len_vel, force0], axis=1)
        return muscle_state0


class RigidTendonHillMuscle(Muscle):
    # --------------------------
    # This is based on Kistemaker et al 2006
    # --------------------------

    def __init__(self, min_activation=0.001, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = 'RigidTendonHillMuscle'

        self.state_name = ['activation',
                           'muscle length',
                           'muscle velocity',
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
        self.k_pe = None
        self.s_as = 0.001
        self.f_iso_n_den = .66 ** 2
        self.b_rel_st_den = 5e-3 - 0.3
        self.k_se = 1 / (0.04 ** 2)
        self.q_crit = 0.3
        self.min_flce = 0.01

        self.to_build_dict = {'timestep': [],
                              'max_isometric_force': [],
                              'tendon_length': [],
                              'optimal_muscle_length': [],
                              'normalized_slack_muscle_length': []}
        self.to_build_dict_default = {'normalized_slack_muscle_length': 1.4}

        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        tendon_length = kwargs.get('tendon_length')
        optimal_muscle_length = kwargs.get('optimal_muscle_length')
        normalized_slack_muscle_length = kwargs.get('normalized_slack_muscle_length')

        self.dt = timestep
        self.n_muscles = np.array(tendon_length).size
        self.l0_ce = tf.reshape(tf.cast(optimal_muscle_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_pe = self.l0_ce * normalized_slack_muscle_length
        self.k_pe = 1 / ((1.66 - self.l0_pe / self.l0_ce) ** 2)
        self.max_iso_force = tf.reshape(tf.cast(max_isometric_force, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_se = tf.reshape(tf.cast(tendon_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.musculotendon_slack_len = self.l0_pe + self.l0_se
        self.vmax = 10 * self.l0_ce
        self.built = True

    def get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        _, muscle0 = self.__call__(excitation=tf.zeros_like(musculotendon_len),
                                   muscle_state=tf.ones_like(musculotendon_len) * self.min_activation,
                                   geometry_state=geometry_state)
        return muscle0

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
        flce = tf.maximum(1 + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den, self.min_flce)

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
        self.__name__ = 'RigidTendonHillMuscleThelen'

        self.state_name = ['activation',
                           'muscle length',
                           'muscle velocity',
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
        self.ce_0 = None
        self.ce_1 = None
        self.ce_2 = None
        self.ce_3 = None
        self.ce_4 = None
        self.ce_5 = None

        self.to_build_dict = {'timestep': [],
                              'max_isometric_force': [],
                              'tendon_length': [],
                              'optimal_muscle_length': [],
                              'normalized_slack_muscle_length': []}
        self.to_build_dict_default = {'normalized_slack_muscle_length': 1.}
        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        tendon_length = kwargs.get('tendon_length')
        optimal_muscle_length = kwargs.get('optimal_muscle_length')
        normalized_slack_muscle_length = kwargs.get('normalized_slack_muscle_length')

        self.n_muscles = np.array(tendon_length).size
        self.dt = timestep
        self.max_iso_force = tf.reshape(tf.cast(max_isometric_force, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_ce = tf.reshape(tf.cast(optimal_muscle_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_pe = self.l0_ce * normalized_slack_muscle_length
        self.l0_se = tf.reshape(tf.cast(tendon_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.musculotendon_slack_len = self.l0_pe + self.l0_se
        self.vmax = 10 * self.l0_ce

        # pre-computed for speed
        self.ce_0 = 3. * self.vmax
        self.ce_1 = self.ce_Af * self.vmax
        self.ce_2 = 3. * self.ce_Af * self.vmax * self.ce_fmlen - 3. * self.ce_Af * self.vmax
        self.ce_3 = 8. * self.ce_Af * self.ce_fmlen + 8. * self.ce_fmlen
        self.ce_4 = self.ce_Af * self.ce_fmlen * self.vmax - self.ce_1
        self.ce_5 = 8. * (self.ce_Af + 1.)

        self.built = True

    def get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        _, muscle0 = self.__call__(excitation=tf.zeros_like(musculotendon_len),
                                   muscle_state=tf.ones_like(musculotendon_len) * self.min_activation,
                                   geometry_state=geometry_state)
        return muscle0

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
                       y=self.ce_2 * new_activation + self.ce_3 * muscle_vel + self.ce_4)
        den = tf.where(condition=muscle_vel <= 0,
                       x=new_activation3 * self.ce_1 + self.ce_1 - 4. * muscle_vel,
                       y=self.ce_4 * new_activation3 + self.ce_5 * muscle_vel + self.ce_4)
        fvce = tf.maximum(nom / den, 0.)
        flpe = tf.maximum((tf.exp(self.pe_1 * (muscle_len - self.l0_pe) / self.l0_ce) - 1) / self.pe_den, 0.)
        flce = tf.exp((- ((muscle_len / self.l0_ce) - 1) ** 2) / self.ce_gamma)

        force = (new_activation * flce * fvce + flpe) * self.max_iso_force
        new_muscle_state = tf.concat([new_activation, muscle_len, muscle_vel, flpe, flce, fvce], axis=1)
        return force, new_muscle_state


class CompliantTendonHillMuscle(Muscle):

    def __init__(self, min_activation=0.01, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = 'CompliantTendonHillMuscle'

        self.state_name = ['activation',
                           'muscle length',
                           'muscle velocity',
                           'force-length PE',
                           'force-length SE',
                           'active force']
        self.state_dim = len(self.state_name)

        # pre-computed for speed
        self.s_as = 0.001
        self.f_iso_n_den = .66 ** 2
        self.b_rel_st_den = 5e-3 - 0.3
        self.k_se = 1 / (0.04 ** 2)

        # pre-define attributes:
        self.musculotendon_slack_len = None
        self.k_pe = None
        self.to_build_dict = {'timestep': [],
                              'max_isometric_force': [],
                              'tendon_length': [],
                              'optimal_muscle_length': [],
                              'normalized_slack_muscle_length': []}
        self.to_build_dict_default = {'normalized_slack_muscle_length': 1.4}
        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        tendon_length = kwargs.get('tendon_length')
        optimal_muscle_length = kwargs.get('optimal_muscle_length')
        normalized_slack_muscle_length = kwargs.get('normalized_slack_muscle_length')

        self.n_muscles = np.array(tendon_length).size
        self.l0_ce = tf.reshape(tf.cast(optimal_muscle_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.l0_pe = self.l0_ce * normalized_slack_muscle_length
        self.l0_se = tf.reshape(tf.cast(tendon_length, dtype=tf.float32), (1, 1, self.n_muscles))
        self.musculotendon_slack_len = self.l0_pe + self.l0_se
        self.k_pe = 1 / ((1.66 - self.l0_pe / self.l0_ce) ** 2)
        self.dt = timestep
        self.max_iso_force = tf.reshape(tf.cast(max_isometric_force, dtype=tf.float32), (1, 1, self.n_muscles))
        self.vmax = 10 * self.l0_ce

    def __call__(self, excitation, muscle_state, geometry_state):
        new_activation = self.activation_ode(excitation, muscle_state)

        # musculotendon geometry
        muscle_len = tf.slice(muscle_state, [0, 1, 0], [-1, 1, -1])
        muscle_len_n = muscle_len / self.l0_ce
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        tendon_len = musculotendon_len - muscle_len
        tendon_strain = tf.maximum((tendon_len - self.l0_se) / self.l0_se, 0.)
        muscle_strain = tf.maximum((muscle_len - self.l0_pe) / self.l0_ce, 0.)

        # forces
        flse = tf.minimum(self.k_se * (tendon_strain ** 2), 3.)
        flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 3.)
        active_force = tf.maximum(flse - flpe, 0.)

        # RK4 integration
        muscle_vel_k1 = self.muscle_ode(muscle_len_n, new_activation, active_force)
        muscle_vel_k2 = self.muscle_ode(muscle_len_n + self.dt * 0.5 * muscle_vel_k1, new_activation, active_force)
        muscle_vel_k3 = self.muscle_ode(muscle_len_n + self.dt * 0.5 * muscle_vel_k2, new_activation, active_force)
        muscle_vel_k4 = self.muscle_ode(muscle_len_n + self.dt * muscle_vel_k3, new_activation, active_force)
        new_muscle_vel_n = (muscle_vel_k1 + 2 * muscle_vel_k2 + 2 * muscle_vel_k3 + muscle_vel_k4) / 6
        new_muscle_vel = new_muscle_vel_n * self.vmax
        new_muscle_len = (muscle_len_n + self.dt * new_muscle_vel_n) * self.l0_ce

        force = flse * self.max_iso_force
        new_muscle_state = tf.concat([new_activation, new_muscle_len, new_muscle_vel, flpe, flse, active_force], axis=1)
        return force, new_muscle_state

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

        # defensive code to avoid propagation of negative square root in the non-selected tf.where outcome
        # the assertion is to ensure that any negative root is indeed a non-selected item.
        sqrt_term = active_force ** 2 - 2 * active_force * p1 * p4 + \
            2 * active_force * p3 + p1 ** 2 * p4 ** 2 - 2 * p1 * p3 * p4 + p2_containing_term + p3 ** 2
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

    def get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        musculotendon_vel = tf.slice(geometry_state, [0, 1, 0], [-1, 1, -1])
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
                    (kpe * lpe * lse ** 2 - kse * lce ** 2 * lmt + kse * lce ** 2 * lse - lce * lse * tf.sqrt(
                        kpe * kse)
                     * (-lmt + lpe + lse)) / (kpe * lse ** 2 - kse * lce ** 2)
                )
            )
        )
        tf.debugging.assert_non_negative(muscle_len, message='initial muscle length was < 0.')
        muscle0_temp = tf.concat((activation, muscle_len, musculotendon_vel), axis=1)
        _, muscle0 = self.__call__(activation, muscle0_temp, geometry_state)
        return muscle0
