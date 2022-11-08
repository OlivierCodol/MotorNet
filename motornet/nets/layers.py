import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, GRUCell, Dense, Lambda
from abc import abstractmethod
from typing import Union
from ..utils import Alias


class Network(Layer):
    """Base class for controller :class:`Network` objects. It implements a network whose function is to control the
    plant provided as input at initialization. This object can be subclassed to implement virtually anything that
    `tensorflow` can implement as a deep neural network, so long as it abides by the state structure used in `motornet`
    (see below for details).

    Args:
        plant: A :class:`motornet.plants.plants.Plant` object class or subclass. This is the plant that the
            :class:`Network` will control.
        proprioceptive_noise_sd: `Float`, the standard deviation of the gaussian noise process for the proprioceptive
            feedback loop. The gaussian noise process is a normal distribution centered on `0`.
        visual_noise_sd: `Float`, the standard deviation of the random noise process for the visual
            feedback loop. The random process is a normal distribution centered on `0`.
        n_ministeps: `Integer`, the number of timesteps that the plant is simulated forward for each forward pass of
            the deep neural network. For instance, if the (global) timestep size is `1` ms, and `n_ministeps` is `5`,
            then the plant will be simulated for every `0.2` ms timesteps, with the excitatory drive from the controller
            only being updated every `1` ms.
        **kwargs: This is passed to the parent `tensorflow.keras.layers.Layer` class as-is.
    """

    def __init__(self, plant, proprioceptive_noise_sd: float = 0., visual_noise_sd: float = 0., n_ministeps: int = 1,
                 **kwargs):

        # set noise levels
        self.proprioceptive_noise_sd = proprioceptive_noise_sd
        self.visual_noise_sd = visual_noise_sd

        # plant states
        self.proprioceptive_delay = plant.proprioceptive_delay
        self.visual_delay = plant.visual_delay
        self.n_muscles = plant.n_muscles
        self.state_size = [
            tf.TensorShape([plant.output_dim]),
            tf.TensorShape([plant.output_dim]),
            tf.TensorShape([plant.muscle_state_dim, self.n_muscles]),
            tf.TensorShape([plant.geometry_state_dim, self.n_muscles]),
            tf.TensorShape([self.n_muscles * 2, self.proprioceptive_delay]),  # muscle length & velocity
            tf.TensorShape([plant.space_dim, self.visual_delay]),
            tf.TensorShape([plant.input_dim]),
        ]
        self.initial_state_names = [
            'joint0',
            'cartesian0',
            'muscle0',
            'geometry0',
            'proprio_feedback0',
            'visual_feedback0',
            'excitation',
        ]
        self.output_names = [
            'joint position',
            'cartesian position',
            'muscle state',
            'geometry state',
            'proprioceptive feedback',
            'visual feedback',
            'excitation'
        ]

        # create attributes
        self.n_ministeps = int(np.maximum(n_ministeps, 1))
        self.output_size = self.state_size
        self.plant = plant
        self.layers = []

        # functionality for recomputing inputs at every timestep
        self.do_recompute_inputs = False
        self.recompute_inputs = lambda inputs, states: inputs

        # create Lambda-wrapped functions (to prevent memory leaks)
        def get_new_proprio_feedback(mstate):
            # normalise by muscle characteristics
            muscle_len = tf.slice(mstate, [0, 1, 0], [-1, 1, -1]) / self.plant.muscle.l0_ce
            muscle_vel = tf.slice(mstate, [0, 2, 0], [-1, 1, -1]) / self.plant.muscle.vmax
            # flatten muscle length and velocity
            proprio_true = tf.reshape(tf.concat([muscle_len, muscle_vel], axis=1), shape=(-1, self.n_muscles * 2))
            return proprio_true

        def get_new_visual_feedback(cstate):
            visual_true, _ = tf.split(cstate, 2, axis=-1)  # position only (discard velocity)
            return visual_true

        name = "get_new_hidden_state"
        self.get_new_hidden_state = Lambda(lambda x: [tf.zeros((x[0], n), dtype=x[1]) for n in self.n_units], name=name)
        self.unpack_plant_states = Lambda(lambda x: x[:4], name="unpack_plant_states")
        self.unpack_feedback_states = Lambda(lambda x: x[4:6], name="unpack_feedback_states")
        self.get_feedback_backlog = Lambda(lambda x: tf.slice(x, [0, 0, 1], [-1, -1, -1]), name="get_feedback_backlog")
        self.get_feedback_current = Lambda(lambda x: x[:, :, 0], name="get_feedback_current")
        self.lambda_cat = Lambda(lambda x: tf.concat(x, axis=-1), name="lambda_cat")
        self.lambda_cat2 = Lambda(lambda x: tf.concat(x, axis=2), name="lambda_cat2")
        self.add_noise = Lambda(lambda x: x[0] + tf.random.normal(tf.shape(x[0]), stddev=x[1]), name="add_noise")
        self.tile_feedback = Lambda(lambda x: tf.tile(x[0][:, :, tf.newaxis], [1, 1, x[1]]), name="tile_feedback")
        self.get_new_proprio_feedback = Lambda(lambda x: get_new_proprio_feedback(x), name="get_new_proprio_feedback")
        self.get_new_visual_feedback = Lambda(lambda x: get_new_visual_feedback(x), name="get_new_visual_feedback")
        self.get_new_excitation_state = Lambda(lambda x: tf.zeros((x[0], self.plant.input_dim), dtype=x[1]))
        self.built = False

        super().__init__(**kwargs)

    state_name = Alias("output_names", alias_name="state_name")
    """An alias name for the `output_names` attribute."""

    @abstractmethod
    def forward_pass(self, inputs, states):
        """Performs the forward pass through the network layers to obtain the motor commands that will then be passed
        on to the plant.

        Args:
            inputs: `Tensor`, inputs to the first layer of the network.
            states: `List` of `tensor` arrays, containing the states of each layer operating on a state.

        Returns:
            - A `tensor` array, the output of the last layer to use as the motor command, or excitation to the plant.
            - A `list` of the new states inherent to potential layers operating on a state.
            - A `dictionary` of the new states inherent to potential layers operating on a state.

        Raises:
            NotImplementedError: If this method is not overwritten by a subclass object.
        """
        raise NotImplementedError("This method must be overwritten by a subclass object.")

    def get_base_config(self):
        """Gets the object instance's base configuration. This is the set of configuration entries that will be useful
        for any :class:`Network` class or subclass. This method should be called by the :meth:`get_save_config`
        method. Users wanting to save additional configuration entries specific to a `Network` subclass should then
        do so in the :meth:`get_save_config` method, using this method's output `dictionary` as a base.

        Returns:
             A `dictionary` containing the network's proprioceptive and visual noise standard deviation and delay, and
             the number of muscles and ministeps.
        """

        cfg = {
            'proprioceptive_noise_sd': self.proprioceptive_noise_sd,
            'visual_noise_sd': self.visual_noise_sd,
            'proprioceptive_delay': self.proprioceptive_delay,
            'visual_delay': self.visual_delay,
            'n_muscle': self.n_muscles,
            'n_ministeps': self.n_ministeps
        }
        return cfg

    def get_save_config(self):
        """Gets the :class:`Network` object's configuration as a `dictionary`. This method should be overwritten by
        subclass objects, and used to add configuration entries specific to that subclass.

        Returns:
            By default, this method returns the output of the :meth:`get_base_config` method.
        """
        return self.get_base_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, states=None, **kwargs):
        """The logic for a single simulation step. This performs a single forward pass through the network, and passes
        the network output as excitation signals (motor commands) to the plant object to simulate movement.

        Args:
            inputs: `Dictionary` of `tensor` arrays. At the very least, this should contain a "inputs" key mapped to
                a `tensor` array, which will be passed as-is to the network's input layer. Additional keys will be
                passed as `**kwargs` to the plant call.
            states: `List`, contains all the states of the plant, and of the network if any exist. The state order in
                the `list` follows the same convention as the :meth:`get_initial_states` method.
            kwargs: For backward compatibility only.

        Returns:
            - A `dictionary` containing the new states.
            - A `list` containing the new states in the same order convention as the :meth:`get_initial_states` method.
              While this output is redundant to the user, it is necessary for `tensorflow` to process the network over
              time.
        """

        # handle feedback
        old_proprio_feedback, old_visual_feedback = self.unpack_feedback_states(states)
        proprio_backlog = self.get_feedback_backlog(old_proprio_feedback)
        visual_backlog = self.get_feedback_backlog(old_visual_feedback)
        proprio_fb = self.get_feedback_current(old_proprio_feedback)
        visual_fb = self.get_feedback_current(old_visual_feedback)

        # if the task demands it, inputs will be recomputed at every timestep
        if self.do_recompute_inputs:
            inputs = self.recompute_inputs(inputs, states)

        x = self.lambda_cat((proprio_fb, visual_fb, inputs.pop("inputs")))
        u, new_network_states, new_network_states_dict = self.forward_pass(x, states)

        # plant forward pass
        jstate, cstate, mstate, gstate = self.unpack_plant_states(states)
        for _ in range(self.n_ministeps):
            jstate, cstate, mstate, gstate = self.plant(u, jstate, mstate, gstate, **inputs)

        proprio_true = self.get_new_proprio_feedback(mstate)
        visual_true = self.get_new_visual_feedback(cstate)
        proprio_noisy = self.add_noise((proprio_true, self.proprioceptive_noise_sd))
        visual_noisy = self.add_noise((visual_true, self.visual_noise_sd))
        new_proprio_feedback = self.lambda_cat2((proprio_backlog, proprio_noisy[:, :, tf.newaxis]))
        new_visual_feedback = self.lambda_cat2((visual_backlog, visual_noisy[:, :, tf.newaxis]))

        # pack new states
        new_states = [jstate, cstate, mstate, gstate, new_proprio_feedback, new_visual_feedback, u]
        new_states.extend(new_network_states)

        # pack output
        output = {
            'joint position': jstate,
            'cartesian position': cstate,
            'muscle state': mstate,
            'geometry state': gstate,
            'proprioceptive feedback': new_proprio_feedback,
            'visual feedback': new_visual_feedback,
            'excitation': u,
            **new_network_states_dict
        }

        return output, new_states

    def get_base_initial_state(self, inputs=None, batch_size: int = 1, dtype=tf.float32):
        """Creates the base initial states for the first timestep of the network training procedure. This method
        provides the base states for the default :class:`Network` class, in the order listed below:

            - joint state
            - cartesian state
            - muscle state
            - geometry state
            - proprioception feedback array
            - visual feedback array
            - excitation state

        This method should be called in the :meth:`get_initial_state` method to provide a base for the output of that
        method.

        Args:
            inputs: The joint state from which the other state values are inferred. This is passed as-is to the
                :meth:`motornet.plants.plants.Plant.get_initial_state` method, and therefore obeys the structure
                documented there.
            batch_size: `Integer`, the batch size defining the size of each state's first dimension.
            dtype: A `dtype` from the `tensorflow.dtypes` module.

        Returns:
            A `list` of the states as `tensor` arrays in the order listed above.
        """

        if inputs is not None:
            states = self.plant.get_initial_state(joint_state=inputs, batch_size=batch_size)
        else:
            states = self.plant.get_initial_state(batch_size=batch_size)

        # no need to add noise as this is just a placeholder for initialization purposes (i.e., not used in first pass)
        excitation = self.get_new_excitation_state((batch_size, dtype))

        proprio_true = self.get_new_proprio_feedback(states[2])
        visual_true = self.get_new_visual_feedback(states[1])
        proprio_tiled = self.tile_feedback((proprio_true, self.proprioceptive_delay))
        visual_tiled = self.tile_feedback((visual_true, self.visual_delay))
        proprio_noisy = self.add_noise((proprio_tiled, self.proprioceptive_noise_sd))
        visual_noisy = self.add_noise((visual_tiled, self.visual_noise_sd))

        states.append(proprio_noisy)
        states.append(visual_noisy)
        states.append(excitation)
        return states

    def get_initial_state(self, inputs=None, batch_size: int = 1, dtype=tf.float32):
        """Creates the initial states for the first timestep of the network training procedure. This method
        provides the states for the full :class:`Network` class, that is the default states from the
        :meth:`get_base_initial_state` method followed by the states specific to a potential subclass. This method
        should be overwritten by subclassing objects, and used to add new states specific to that subclass.

        Args:
            inputs: The joint state from which the other state values are inferred. This is passed as-is to the
                :meth:`motornet.plants.plants.Plant.get_initial_state` method, and therefore obeys the logic
                documented there.
            batch_size: `Integer`, the batch size defining the size of each state's first dimension.
            dtype: A `dtype` from the `tensorflow.dtypes` module.

        Returns:
            By default, this method returns the output of the :meth:`get_base_initial_state` method, that is a
            `list` of the states as `tensor` arrays.
        """
        return self.get_base_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)


class GRUNetwork(Network):
    """A GRU network acting as a controller to the plant. The last layer of the network is a
    `tensorflow.keras.layers.Dense` dense layer containing `n_muscles * k` units, with `k` being the number of inputs
    a single muscle takes (as defined by the `muscle_type` that the plant object uses), and `n_muscles` the number of
    muscles that the plant contains.

    Args:
        plant: A :class:`motornet.plants.plants.Plant` object class or subclass. This is the plant that the `Network`
            will control.
        n_units: `Integer` or `list`, the number of GRUs per layer. If only one layer is created, then this can
            be an `integer`.
        n_hidden_layers: `Integer`, the number of hidden layers of GRUs that the network will implement.
        activation: `String` or activation function from `tensorflow`. The activation function used as non-linearity
            for all GRUs.
        kernel_regularizer: `Float`, the kernel regularization weight for the GRUs.
        recurrent_regularizer: `Float`, the recurrent regularization weight for the GRUs.
        hidden_noise_sd: `Float`, the standard deviation of the gaussian noise process applied to GRU hidden activity.
        output_bias_initializer: A `tensorflow.keras.initializers` instance to initialize the biases of the
            last layer of the network (`i.e.`, the output layer).
        output_kernel_initializer: A `tensorflow.keras.initializers` instance to initialize the kernels of the
            last layer of the network (`i.e.`, the output layer).
        **kwargs: This is passed to the parent `tensorflow.keras.layers.Layer` class as-is.
    """

    def __init__(self, plant, n_units: Union[int, list] = 20, n_hidden_layers: int = 1, activation='tanh',
                 kernel_regularizer: float = 0., recurrent_regularizer: float = 0., hidden_noise_sd: float = 0.,
                 output_bias_initializer=tf.initializers.Constant(value=-5),
                 output_kernel_initializer=tf.keras.initializers.random_normal(stddev=10 ** -3), **kwargs):

        super().__init__(plant, **kwargs)

        if type(n_units) == int:
            n_units = list(np.repeat(n_units, n_hidden_layers).astype('int32'))
        if len(n_units) > 1 and n_hidden_layers == 1:
            n_hidden_layers = len(n_units)
        if len(n_units) != n_hidden_layers:
            raise ValueError('The number of hidden layers should match the size of the n_unit array.')

        # set noise levels
        self.hidden_noise_sd = hidden_noise_sd

        # hidden states for GRU layer(s)
        self.n_units = n_units
        self.n_hidden_layers = n_hidden_layers
        self.layer_state_names = ['gru_hidden_' + str(k) for k in range(self.n_hidden_layers)]
        self.output_names.extend(self.layer_state_names)
        self.initial_state_names.extend([name + '_0' for name in self.layer_state_names])

        for n in n_units:
            self.state_size.append(tf.TensorShape([n]))

        # create attributes
        self.kernel_regularizer_weight = kernel_regularizer  # to save the values in `get_save_config`
        self.kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        self.recurrent_regularizer_weight = recurrent_regularizer  # to save the values in `get_save_config`
        self.recurrent_regularizer = tf.keras.regularizers.l2(recurrent_regularizer)
        self.output_bias_initializer = output_bias_initializer
        self.output_kernel_initializer = output_kernel_initializer

        if activation == 'recttanh':
            self.activation = recttanh
            self.activation_name = 'recttanh'
        else:
            self.activation = activation
            self.activation_name = activation

    def build(self, input_shapes):

        for k in range(self.n_hidden_layers):
            layer = GRUCell(
                units=self.n_units[k],
                activation=self.activation,
                name='hidden_layer_' + str(k),
                kernel_regularizer=self.kernel_regularizer,
                recurrent_regularizer=self.recurrent_regularizer,
            )
            self.layers.append(layer)

        output_layer = Dense(
            units=self.plant.input_dim,
            activation='sigmoid',
            name='output_layer',
            bias_initializer=self.output_bias_initializer,
            kernel_initializer=self.output_kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        self.layers.append(output_layer)
        self.built = True

    def get_initial_state(self, inputs=None, batch_size: int = 1, dtype=tf.float32):
        """Creates the initial states for the first timestep of the network training procedure. This method
        provides the states for the full :class:`Network` class, that is the default states from the
        :meth:`Network.get_base_initial_state` method followed by the states specific to this subclass.

        Args:
            inputs: The joint state from which the other state values are inferred. This is passed as-is to the
                :meth:`motornet.plants.plants.Plant.get_initial_state` method, and therefore obeys the structure documented
                there.
            batch_size: `Integer`, the batch size defining the size of each state's first dimension.
            dtype: A `dtype` from the `tensorflow.dtypes` module.

        Returns:
            A `list` containing the output of the :meth:`Network.get_base_initial_state` method, followed by the hidden states
            of each GRU layer as `tensor` arrays.
        """
        states = self.get_base_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)
        hidden_states = self.get_new_hidden_state((batch_size, dtype))
        states.extend(hidden_states)
        return states

    def get_save_config(self):
        """Gets the base configuration from the :meth:`motornet.nets.layers.Network.get_base_config` method, and adds
        the configuration information specific to the :class:`GRUNetwork` class to that `dictionary`. These are:

            - The standard deviation of the gaussian noise process to the GRU hidden activity.
            - The kernel regularizer weight.
            - The recurrent regularizer weight.
            - The number of GRU units per hidden layer. If there are several layers, this will be a `list`.
            - The number of hidden GRU layers.
            - The name of the activation function used as non-linearity for the hidden GRU layers.

        Returns:
            A `dictionary` containing the object instance's full configuration.
        """

        base_config = self.get_base_config()
        cfg = {
            'hidden_noise_sd': self.hidden_noise_sd,
            'kernel_regularizer_weight': self.kernel_regularizer_weight,
            'recurrent_regularizer_weight': self.recurrent_regularizer_weight,
            'n_units': int(self.n_units[0]),
            'n_hidden_layers': self.n_hidden_layers,
            'activation': self.activation_name, **base_config
        }
        return cfg

    def forward_pass(self, inputs, states):
        """Performs the forward pass computation.

        Args:
            inputs: `Tensor`, inputs to the first layer of the network.
            states: `List` of `tensor` arrays representing the states of each layer operating on a state (state-based
                layers).

        Returns:
            - A `tensor` array, the output of the last layer to use as the motor command, or excitation to the plant.
            - A `list` of the new hidden states of the GRU layers.
            - A `dictionary` of the new hidden states of the GRU layers.
        """
        new_hidden_states_dict = {}
        new_hidden_states = []
        x = inputs

        for k in range(self.n_hidden_layers):
            x, new_hidden_state = self.layers[k](x, states[- self.n_hidden_layers + k])
            new_hidden_state_noisy = self.add_noise((new_hidden_state, self.hidden_noise_sd))
            new_hidden_states_dict[self.layer_state_names[k]] = new_hidden_state_noisy
            new_hidden_states.append(new_hidden_state_noisy)
        u = self.layers[-1](x)
        return u, new_hidden_states, new_hidden_states_dict


@tf.function
def recttanh(x):
    """A rectified hyperbolic tangent activation function."""
    x = tf.keras.activations.tanh(x)
    x = tf.where(tf.less_equal(x, tf.constant(0.)), tf.constant(0.), x)
    return x
