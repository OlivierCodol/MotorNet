import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from abc import abstractmethod


class Muscle:
    """Base class for `Muscle` objects. If a plant contains several muscles, this object will contain all of
    those in a vectorized format, meaning for any given plant there will always be only `one` muscle object, regardless
    of the number of muscles wrapped around the skeleton.

    The dimensionality of the muscle states produced by this object and subclasses will always be
    `n_batches * n_timesteps * n_states * n_muscles`.

    Args:
        input_dim: `Integer`, the dimensionality of the drive input to the muscle. For instance, if the muscle is only
            driven by an excitation signal, then this value should be `1`.
        output_dim: `Integer`, the dimensionality of the output. Since this object does not rely on a ``call`` method,
            but on an :meth:`integrate` method, this is the dimensionality of the :meth:`integrate` method's output.
            The output of that method should be a `muscle state`, so `output_dim` is usually the number of states
            of the :class:`Muscle` object class or subclass.
        min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
            this value will be clipped.
        tau_activation: `Float`, the time constant for activation of the muscle. This is used for the Ordinary
            Differential Equation of the muscle activation method :meth:`activation_ode`.
        tau_deactivation: `Float`, the time constant for deactivation of the muscle. This is used for the Ordinary
            Differential Equation of the muscle activation method :meth:`activation_ode`.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1, min_activation: float = 0.,
                 tau_activation: float = 0.015, tau_deactivation: float = 0.05):
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
        self._get_initial_muscle_state_fn = Lambda(lambda x: self._get_initial_muscle_state(*x), name='get_init_mstate')
        self._activation_ode_fn = Lambda(lambda x: self._activation_ode(*x), name='activation_ode')
        self.clip_activation = Lambda(lambda a: tf.clip_by_value(a, self.min_activation, 1.), name='clip_activation')
        self._integrate_fn = Lambda(lambda x: self._integrate(*x), name='muscle_integrate')
        self._update_ode_fn = Lambda(lambda x: self._update_ode(*x), name='muscle_update_ode')
        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        """Build the muscle given parameters from the ``motornet.plants.Plant`` wrapper object. This should be called by
        the :meth:`motornet.plants.Plant.add_muscle` method to build the muscle scructure according to the parameters of
        that plant.

        Args:
            timestep: `Float`, the size of a single timestep in seconds.
            max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
                muscle can use. If several muscles are being built, then this should be a `list` containing as many
                elements as there are muscles.
            **kwargs: Optional keyword arguments. This allows for extra parameters to be passed in the :meth:`build`
                method for a :class:`Muscle` subclass, if needed.
        """
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
        """Computes the new activation value of the (set of) muscle(s) according to the Ordinary Differential Equation
        shown in equations 1-2 in `[1]`.

        References:
            [1] `Thelen DG. Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older
            adults. J Biomech Eng. 2003 Feb;125(1):70-7. doi: 10.1115/1.1531112. PMID: 12661198.`

        Args:
            excitation: `Float` or `list` of `float`, the descending excitation drive to the muscle(s). If several
                muscles are declared in the parent plant object, then this should be a `list` containing as many
                elements as there are muscles in that parent plant object.
            muscle_state: `Tensor`, the `muscle state` that provides the initial activation value for the Ordinary
                Differential Equation.

        Returns:
            A `tensor` containing the updated activation values.
        """
        return self._activation_ode_fn((excitation, muscle_state))

    def get_initial_muscle_state(self, batch_size: int, geometry_state):
        """Infers the `muscle state` matching a provided `geometry state` array.

        Args:
            batch_size: `Integer`, the size of the batch passed in `geometry state`.
            geometry_state: `Tensor`, the `geometry state` array from which the matching initial `muscle state` is
                inferred.

        Returns:
            A `tensor` containing the initial `muscle state` matching the input `geometry state` array.
        """
        return self._get_initial_muscle_state_fn((batch_size, geometry_state))

    def integrate(self, dt, state_derivative, muscle_state, geometry_state):
        """Performs one integration step for the muscle step.

        Args:
            dt: `Float`, size of the timestep in seconds for this integration step.
            state_derivative: `Tensor`, the derivatives of the `muscle state`. These are usually obtained using this
                object's :meth:`update_ode` method.
            muscle_state: `Tensor`, the `muscle state` used as the initial state value for the numerical integration.
            geometry_state: `Tensor`, the `geometry state` used as the initial state value for the numerical integration.

        Returns:
            A `tensor` containing the new `muscle state` following numerical integration.
        """
        return self._integrate_fn((dt, state_derivative, muscle_state, geometry_state))

    def update_ode(self, excitation, muscle_state):
        """Computes the derivatives of `muscle state` using the corresponding Ordinary Differential Equations.

        Args:
            excitation: `Float` or `list` of `float`, the descending excitation drive to the muscle(s). If several
                muscles are declared in the parent plant object, then this should be a `list` containing as many
                elements as there are muscles in that parent plant object.
            muscle_state: `Tensor`, the `muscle state` used as the initial value for the evaluation of the Ordinary
                Differential Equations.

        Returns:
            A `tensor` containing the derivatives of the `muscle state`.
        """
        return self._update_ode_fn((excitation, muscle_state))

    @abstractmethod
    def _get_initial_muscle_state(self, batch_size, geometry_state):
        return

    @abstractmethod
    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        return

    def _update_ode(self, excitation, muscle_state):
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1])
        return self.activation_ode(excitation, activation)

    def _activation_ode(self, excitation, activation):
        excitation = self.clip_activation(tf.reshape(excitation, (-1, 1, self.n_muscles)))
        activation = self.clip_activation(activation)
        tmp = 0.5 + 1.5 * activation
        tau = tf.where(excitation > activation, self.tau_activation * tmp, self.tau_deactivation / tmp)
        return (excitation - activation) / tau

    def setattr(self, name: str, value):
        """Changes the value of an attribute held by this object.

        Args:
            name: `String`, attribute to set to a new value.
            value: Value that the attribute should take.
        """
        self.__setattr__(name, value)

    def get_save_config(self):
        """Gets the object instance's configuration. This is the set of configuration entries that will be useful
        for any muscle objects or subclasses.

        Returns:
             - A `dictionary` containing the muscle object's name and state names.
        """
        cfg = {'name': str(self.__name__), 'state names': self.state_name}
        return cfg

# A "rectified linear" muscle that outputs the input directly, but can only have a positive activation value.
# In essence, this muscle object behaves as a simple linear actuator that can only pull, and not push. It does not
# have an upper bound and its output (assuming it is positive) is scaled by the declared maximum isometric force.
# Therefore, an input of `1` will yield an output force equal to the maximum isometric force.


class ReluMuscle(Muscle):
    """A "rectified linear" muscle whose force output :math:`F` is a linear function of its activation value, which
    itself is bounded between `0` and `1`. Specifically:

    .. math::
        F = m * activation

    with :math:`m` the maximum isometric force of the muscle. Note that the maximum isometric force is not declared at
    initialization but via the :meth:`Muscle.build` call, which is inherited from the parent :class:`Muscle` class.
    The :math:`activation` value is the result of an Ordinary Differential Equation computed by the
    :meth:`Muscle.activation_ode` method. It is not directly the `excitation` input drive.

    Args:
        **kwargs: All contents are passed to the parent :class:`Muscle` class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = 'ReluMuscle'

        self.state_name = ['activation',
                           'muscle length',
                           'muscle velocity',
                           'force']
        self.state_dim = len(self.state_name)

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1]) + state_derivative * dt
        activation = self.clip_activation(activation)
        forces = tf.maximum(activation, 0.) * self.max_iso_force
        muscle_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_vel = tf.slice(geometry_state, [0, 1, 0], [-1, 1, -1])
        return tf.concat([activation, muscle_len, muscle_vel, forces], axis=1)

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        excitation0 = tf.ones((batch_size, 1, self.n_muscles)) * self.min_activation
        force0 = tf.zeros((batch_size, 1, self.n_muscles))
        len_vel = tf.slice(geometry_state, [0, 0, 0], [-1, 2, -1])
        return tf.concat([excitation0, len_vel, force0], axis=1)


class RigidTendonHillMuscle(Muscle):
    """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`, adjusted to
    behave as a rigid tendon version of the original model.

    References:
        [1] `Kistemaker DA, Wong JD, Gribble PL. The central nervous system does not minimize energy cost in arm
        movements. J Neurophysiol. 2010 Dec;104(6):2985-94. doi: 10.1152/jn.00483.2010. Epub 2010 Sep 8. PMID:
        20884757.`

    Args:
        min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
            this value will be clipped.
        **kwargs: All contents are passed to the parent :class:`Muscle` class.
    """

    def __init__(self, min_activation=0.001, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = 'RigidTendonHillMuscle'

        self.state_name = [
            'activation',
            'muscle length',
            'muscle velocity',
            'force-length PE',
            'force-length CE',
            'force-velocity CE',
            'force'
        ]
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
        self.to_build_dict_default = {'normalized_slack_muscle_length': 1.}

        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        """Build the muscle using arguments from the :class:`motornet.plants.plants.Plant` wrapper object. This should
        be called by the :meth:`motornet.plants.plants.Plant.add_muscle` method to build the muscle structure according
        to the parameters of that plant.

        Args:
            timestep: `Float`, the size of a single timestep in seconds.
            max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
                muscle can use. If several muscles are being built, then this should be a list containing as many
                elements as there are muscles.
            tendon_length: `Float` or `list` of `float`, the tendon length (m) of the muscle(s). If several
                muscles are declared in the parent plant object, then this should be a list containing as many elements
                as there are muscles in that parent plant object.
            optimal_muscle_length: `Float` or `list` of `float`, the optimal length (m) of the muscle(s). This defines
                the length at which the muscle will output the maximum amount of force given the same excitation. If
                several muscles are declared in the parent plant object, then this should be a list containing as many
                elements as there are muscles in that parent plant object.
            normalized_slack_muscle_length: `Float` or `list` of `float`, the muscle length (m) past which the
                muscle(s) will start to developp passive forces. If several muscles are declared in the parent plant
                object, then this should be a list containing as many elements as there are muscles in that parent plant
                object.
        """
        tendon_length = kwargs.get('tendon_length')
        optimal_muscle_length = kwargs.get('optimal_muscle_length')
        normalized_slack_muscle_length = kwargs.get('normalized_slack_muscle_length')

        self.n_muscles = np.array(tendon_length).size
        shape = (1, 1, self.n_muscles)

        self.dt = tf.constant(timestep, name='dt')
        self.max_iso_force = tf.reshape(tf.constant(max_isometric_force, dtype=tf.float32, name='max_iso_force'), shape)
        self.l0_ce = tf.reshape(tf.constant(optimal_muscle_length, dtype=tf.float32, name='l0_ce'), shape)
        self.l0_se = tf.reshape(tf.constant(tendon_length, dtype=tf.float32, name='l0_se'), shape)
        self.l0_pe = tf.constant(self.l0_ce * normalized_slack_muscle_length, name='l0_pe')
        self.k_pe = tf.constant(1 / ((1.66 - self.l0_pe / self.l0_ce) ** 2), name='k_pe')
        self.musculotendon_slack_len = tf.constant(self.l0_pe + self.l0_se, name='musculotendon_slack_len')
        self.vmax = tf.constant(10 * self.l0_ce, name='vmax')
        self.built = True

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_state = tf.ones_like(musculotendon_len) * self.min_activation
        return self.integrate(self.dt, tf.zeros_like(musculotendon_len), muscle_state, geometry_state)

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = self.clip_activation(tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1]) + state_derivative * dt)

        # musculotendon geometry
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_vel = tf.slice(geometry_state, [0, 1, 0], [-1, 1, -1])
        muscle_len = tf.maximum(musculotendon_len - self.l0_se, 0.)
        muscle_strain = tf.maximum((muscle_len - self.l0_pe) / self.l0_ce, 0.)
        muscle_len_n = muscle_len / self.l0_ce
        muscle_vel_n = muscle_vel / self.vmax

        # muscle forces
        # flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 3.)
        flpe = self.k_pe * (muscle_strain ** 2)
        flce = tf.maximum(1 + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den, self.min_flce)

        a_rel_st = tf.where(muscle_len_n > 1., .41 * flce, .41)
        b_rel_st = tf.where(
            condition=activation < self.q_crit,
            x=5.2 * (1 - .9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
            y=5.2)
        dfdvcon0 = activation * (flce + a_rel_st) / b_rel_st  # inv of slope at isometric point wrt concentric curve

        f_x_a = flce * activation  # to speed up computation

        tmp_p_nom = f_x_a * .5
        tmp_p_den = self.s_as - dfdvcon0 * 2.

        p1 = - tmp_p_nom / tmp_p_den
        p2 = (tmp_p_nom ** 2) / tmp_p_den
        p3 = - 1.5 * f_x_a

        nom = tf.where(
            condition=muscle_vel_n < 0,
            x=muscle_vel_n * activation * a_rel_st + f_x_a * b_rel_st,
            y=-p1 * p3 + p1 * self.s_as * muscle_vel_n + p2 - p3 * muscle_vel_n + self.s_as * muscle_vel_n ** 2)
        den = tf.where(condition=muscle_vel_n < 0, x=b_rel_st - muscle_vel_n, y=p1 + muscle_vel_n)

        active_force = tf.maximum(nom / den, 0.)
        force = (active_force + flpe) * self.max_iso_force
        return tf.concat([activation, muscle_len, muscle_vel, flpe, flce, active_force, force], axis=1)


class RigidTendonHillMuscleThelen(Muscle):
    """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`, adjusted to
    behave as a rigid tendon version of the original model.

    References:
        [1] `Thelen DG. Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older
        adults. J Biomech Eng. 2003 Feb;125(1):70-7. doi: 10.1115/1.1531112. PMID: 12661198.`

    Args:
        min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
            this value will be clipped.
        **kwargs: All contents are passed to the parent :class:`Muscle` class.
    """

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
        self.built = False

    def build(self, timestep, max_isometric_force, **kwargs):
        """Build the muscle using arguments from the :class:`motornet.plants.plants.Plant` wrapper object. This should
        be called by the :meth:`motornet.plants.plants.Plant.add_muscle` method to build the muscle scructure according
        to the parameters of that plant.

        Args:
            timestep: `Float`, the size of a single timestep in seconds.
            max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
                muscle can use. If several muscles are being built, then this should be a list containing as many
                elements as there are muscles.
            tendon_length: `Float` or `list` of `float`, the tendon length (m) of the muscle(s). If several
                muscles are declared in the parent plant object, then this should be a list containing as many elements
                as there are muscles in that parent plant object.
            optimal_muscle_length: `Float` or `list` of `float`, the optimal length (m) of the muscle(s). This defines
                the length at which the muscle will output the maximum amount of force given the same excitation. If
                several muscles are declared in the parent plant object, then this should be a list containing as many
                elements as there are muscles in that parent plant object.
            normalized_slack_muscle_length: `Float` or `list` of `float`, the muscle length (m) past which the
                muscle(s) will start to developp passive forces. If several muscles are declared in the parent plant
                object, then this should be a list containing as many elements as there are muscles in that parent plant
                object.
        """

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
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_state = tf.ones_like(musculotendon_len) * self.min_activation
        return self.integrate(self.dt, tf.zeros_like(musculotendon_len), muscle_state, geometry_state)

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1]) + state_derivative * dt
        activation = self.clip_activation(activation)

        # musculotendon geometry
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        muscle_len = tf.maximum(musculotendon_len - self.l0_se, 0.001)
        muscle_vel = tf.slice(geometry_state, [0, 1, 0], [-1, 1, -1])

        # muscle forces
        a3 = activation * 3.
        nom = tf.where(condition=muscle_vel <= 0,
                       x=self.ce_Af * (activation * self.ce_0 + 4. * muscle_vel + self.vmax),
                       y=self.ce_2 * activation + self.ce_3 * muscle_vel + self.ce_4)
        den = tf.where(condition=muscle_vel <= 0,
                       x=a3 * self.ce_1 + self.ce_1 - 4. * muscle_vel,
                       y=self.ce_4 * a3 + self.ce_5 * muscle_vel + self.ce_4)
        fvce = tf.maximum(nom / den, 0.)
        flpe = tf.maximum((tf.exp(self.pe_1 * (muscle_len - self.l0_pe) / self.l0_ce) - 1) / self.pe_den, 0.)
        flce = tf.exp((- ((muscle_len / self.l0_ce) - 1) ** 2) / self.ce_gamma)
        force = (activation * flce * fvce + flpe) * self.max_iso_force
        return tf.concat([activation, muscle_len, muscle_vel, flpe, flce, fvce, force], axis=1)


class CompliantTendonHillMuscle(RigidTendonHillMuscle):
    """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`. Unlike its
    parent class, this class implements a full compliant tendon version of the model, as formulated in the
    reference article.

    References:
        [1] `Kistemaker DA, Wong JD, Gribble PL. The central nervous system does not minimize energy cost in arm
        movements. J Neurophysiol. 2010 Dec;104(6):2985-94. doi: 10.1152/jn.00483.2010. Epub 2010 Sep 8. PMID:
        20884757.`

    Args:
        min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
            this value will be clipped.
        **kwargs: All contents are passed to the parent :class:`Muscle` class.
    """

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
        self._muscle_ode_fn = Lambda(lambda x: self._muscle_ode_lambda(*x), name='muscle_ode')
        self.built = False

    def _muscle_ode(self, muscle_len_n, activation, active_force):
        """This wrapper allows to keep track of the argument names to be passed through the lambda function."""
        return self._muscle_ode_fn((muscle_len_n, activation, active_force))

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        # Compute musculotendon geometry
        muscle_len = tf.slice(muscle_state, [0, 1, 0], [-1, 1, -1])
        muscle_len_n = muscle_len / self.l0_ce
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        tendon_len = musculotendon_len - muscle_len
        tendon_strain = tf.maximum((tendon_len - self.l0_se) / self.l0_se, 0.)
        muscle_strain = tf.maximum((muscle_len - self.l0_pe) / self.l0_ce, 0.)

        # Compute forces
        flse = tf.minimum(self.k_se * (tendon_strain ** 2), 1.)
        # flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 1.)
        flpe = self.k_pe * (muscle_strain ** 2)
        active_force = tf.maximum(flse - flpe, 0.)

        # Integrate
        d_activation = tf.slice(state_derivative, [0, 0, 0], [-1, 1, -1])
        muscle_vel_n = tf.slice(state_derivative, [0, 1, 0], [-1, 1, -1])
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1]) + d_activation * dt
        activation = self.clip_activation(activation)
        new_muscle_len = (muscle_len_n + dt * muscle_vel_n) * self.l0_ce

        muscle_vel = muscle_vel_n * self.vmax
        force = flse * self.max_iso_force
        return tf.concat([activation, new_muscle_len, muscle_vel, flpe, flse, active_force, force], axis=1)

    def _update_ode(self, excitation, muscle_state):
        activation = tf.slice(muscle_state, [0, 0, 0], [-1, 1, -1])
        d_activation = self.activation_ode(excitation, activation)
        muscle_len_n = tf.slice(muscle_state, [0, 1, 0], [-1, 1, -1]) / self.l0_ce
        active_force = tf.slice(muscle_state, [0, 5, 0], [-1, 1, -1])
        new_muscle_vel_n = self._muscle_ode(muscle_len_n, activation, active_force)
        return tf.concat([d_activation, new_muscle_vel_n], axis=1)

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = tf.slice(geometry_state, [0, 0, 0], [-1, 1, -1])
        activation = tf.ones_like(musculotendon_len) * self.min_activation

        # if musculotendon length is negative, raise an error.
        # if musculotendon length is less than tendon slack length, assign all (most of) the length to the tendon.
        # if musculotendon length is more than tendon slack length and less than musculotendon slack length, assign to
        #   the tendon up to the tendon slack length, and the rest to the muscle length.
        # if musculotendon length is more than tendon slack length and muscle slack length combined, find the muscle
        #   length that satisfies equilibrium between tendon passive forces and muscle passive forces.
        muscle_len = tf.where(
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
                      (self.k_pe * self.l0_se ** 2 - self.k_se * self.l0_ce ** 2))))

        # tf.debugging.assert_non_negative(muscle_len, message='initial muscle length was < 0.')
        tendon_len = musculotendon_len - muscle_len
        tendon_strain = tf.maximum((tendon_len - self.l0_se) / self.l0_se, 0.)
        muscle_strain = tf.maximum((muscle_len - self.l0_pe) / self.l0_ce, 0.)

        # Compute forces
        flse = tf.minimum(self.k_se * (tendon_strain ** 2), 1.)
        flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 1.)
        active_force = tf.maximum(flse - flpe, 0.)

        muscle_vel_n = self._muscle_ode(muscle_len / self.l0_ce, activation, active_force)
        muscle_state = tf.concat([activation, muscle_len], axis=1)
        state_derivative = tf.concat([tf.zeros_like(musculotendon_len), muscle_vel_n], axis=1)

        return self.integrate(self.dt, state_derivative, muscle_state, geometry_state)

    def _muscle_ode_lambda(self, muscle_len_n, activation, active_force):
        flce = tf.maximum(1. + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den, self.min_flce)
        a_rel_st = tf.where(muscle_len_n > 1., .41 * flce, .41)
        b_rel_st = tf.where(
            condition=activation < self.q_crit,
            x=5.2 * (1 - .9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
            y=5.2)
        # inv of slope at isometric point wrt concentric curve
        f_x_a = flce * activation  # to speed up computation
        dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

        p1 = - f_x_a * .5 / (self.s_as - dfdvcon0 * 2.)
        p3 = - 1.5 * f_x_a
        p2_containing_term = (4 * ((f_x_a * 0.5) ** 2) * (- self.s_as)) / (self.s_as - dfdvcon0 * 2)

        # defensive code to avoid propagation of negative square root in the non-selected tf.where outcome
        # the assertion is to ensure that any selected item is indeed not a negative root.
        sqrt_term = active_force ** 2 + 2 * active_force * p1 * self.s_as + \
            2 * active_force * p3 + p1 ** 2 * self.s_as ** 2 + 2 * p1 * p3 * self.s_as +\
            p2_containing_term + p3 ** 2
        cond = tf.where(tf.logical_and(sqrt_term < 0, active_force >= f_x_a), -1, 1)
        tf.debugging.assert_non_negative(cond, message='root that should be used is negative.')
        sqrt_term = tf.maximum(sqrt_term, 0.)

        new_muscle_vel_nom = tf.where(
            condition=active_force < f_x_a,
            x=b_rel_st * (active_force - f_x_a),
            y=- active_force + p1 * self.s_as - p3 - tf.sqrt(sqrt_term))
        new_muscle_vel_den = tf.where(
            condition=active_force < f_x_a,
            x=active_force + activation * a_rel_st,
            y=- 2 * self.s_as)

        return new_muscle_vel_nom / new_muscle_vel_den
