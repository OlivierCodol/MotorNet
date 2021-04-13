import numpy as np
import tensorflow as tf
# TODO add noise option to each muscle?


class Muscle:
    # --------------------------
    # base class for muscles
    # --------------------------
    def __init__(self, input_dim=1, state_dim=1, output_dim=1, timestep=.01, min_activation=0.):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.state_name = []
        self.output_dim = output_dim
        self.dt = timestep
        self.min_activation = min_activation
        self.to_build_dict = {'timestep': []}
        self.built = False

    def setattr(self, name: str, value):
        self.__setattr__(name, value)

    def build(self, **kwargs):
        self.dt = kwargs.get('timestep', 0.01)
        self.built = True


class ReluMuscle(Muscle):
    # --------------------------
    # A rectified linear muscle that outputs the input directly, but can only have a positive activation value.
    # --------------------------
    def __init__(self, timestep=0.01, min_activation=0., **kwargs):
        super().__init__(timestep=timestep, min_activation=min_activation, **kwargs)
        self.state_name = ['dummy_state']
        self.state_dim = len(self.state_name)

    def __call__(self, excitation, *args, **kwargs):
        return {'forces': tf.nn.relu(excitation), 'muscle_state': tf.zeros_like(excitation)}

    @staticmethod
    def get_initial_muscle_state(batch_size, geometry):
        n_muscles = tf.shape(geometry)[-1]
        return tf.zeros((batch_size, 1, n_muscles))



class RigidTendonHillMuscle(Muscle):
    # --------------------------
    # This is based on Thelen et al 2003
    # --------------------------
    def __init__(self, tau_activation=0.015, tau_deactivation=0.05, min_activation=0.001, timestep=0.01, **kwargs):

        super().__init__(timestep=timestep, min_activation=min_activation, **kwargs)

        self.state_name = ['activation',
                           'muscle length',
                           'muscle_velocity',
                           'force-length PE',
                           'force-length CE',
                           'force-velocity CE']
        self.state_dim = len(self.state_name)

        # parameters for the 1st-order ODE of muscle activation
        self.tau_activation = tau_activation
        self.tau_deactivation = tau_deactivation

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

        self.to_build_dict = {'max_isometric_force': [],
                              'tendon_length': [],
                              'optimal_muscle_length': [],
                              'timestep': []}
        self.built = False

    def build(self, max_isometric_force, tendon_length, optimal_muscle_length, timestep=0.01):
        self.n_muscles = np.array(tendon_length).size
        self.dt = timestep
        self.max_iso_force = tf.reshape(tf.cast(max_isometric_force, dtype=tf.float32), (-1, 1, self.n_muscles))
        self.l0_se = tf.reshape(tf.cast(tendon_length, dtype=tf.float32), (-1, 1, self.n_muscles))
        self.l0_ce = tf.reshape(tf.cast(optimal_muscle_length, dtype=tf.float32), (-1, 1, self.n_muscles))
        self.l0_pe = self.l0_ce * 1.4
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
        # compute muscle activations
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1])
        excitation = tf.reshape(excitation, (-1, 1, self.n_muscles))
        activation = tf.clip_by_value(activation, self.min_activation, 1.)
        excitation = tf.clip_by_value(excitation, self.min_activation, 1.)

        tau_scaler = 0.5 + 1.5 * activation
        tau = tf.where(excitation > activation, self.tau_activation * tau_scaler, self.tau_deactivation / tau_scaler)
        d_activation = (excitation - activation) / tau
        new_activation = activation + d_activation * self.dt
        new_activation = tf.clip_by_value(new_activation, self.min_activation, 1.)

        # musculotendon geometry
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_len = tf.maximum(musculotendon_len - self.l0_se, 0.)
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
        new_muscle_state = tf.concat([new_activation, muscle_len, muscle_vel, force_len_pe, force_len_ce, force_vel_ce], axis=1)
        return force, new_muscle_state
