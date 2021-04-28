import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, GRUCell, Dense


class GRUController(Layer):
    def __init__(self, plant, n_units=20, n_hidden_layers=1, activation='tanh', kernel_regularizer=0.,
                 activity_regularizer=0., recurrent_regularizer=0., proprioceptive_noise_sd=0., visual_noise_sd=0.,
                 perturbation_dim_start=None, **kwargs):

        if type(n_units) == int:
            n_units = list(np.repeat(n_units, n_hidden_layers).astype('int32'))

        # set feedback noise levels
        self.proprioceptive_noise_sd = proprioceptive_noise_sd
        self.visual_noise_sd = visual_noise_sd

        # plant states
        self.proprioceptive_delay = plant.proprioceptive_delay
        self.visual_delay = plant.visual_delay
        self.n_muscles = plant.n_muscles
        self.state_size = [tf.TensorShape([plant.output_dim]),
                           tf.TensorShape([plant.output_dim]),
                           tf.TensorShape([plant.muscle_state_dim, self.n_muscles]),
                           tf.TensorShape([plant.geometry_state_dim, self.n_muscles]),
                           tf.TensorShape([self.n_muscles * 2, self.proprioceptive_delay]),  # muscle length & velocity
                           tf.TensorShape([plant.space_dim, self.visual_delay])]
        # hidden states for GRU layer(s)
        for n in n_units:
            self.state_size.append(tf.TensorShape([n]))

        # set perturbation dimensions of input
        self.perturbation_dim_start = perturbation_dim_start

        self.output_size = self.state_size
        self.plant = plant
        self.kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        self.activity_regularizer = tf.keras.regularizers.l2(activity_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.l2(recurrent_regularizer)
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.n_units = n_units
        self.layers = []
        self.built = False
        super().__init__(**kwargs)

    def build(self, input_shapes):
        for k in range(self.n_hidden_layers):
            layer = GRUCell(units=self.n_units[k],
                            activation=self.activation,
                            name='hidden_layer_' + str(k),
                            kernel_regularizer=self.kernel_regularizer,
                            activity_regularizer=self.activity_regularizer,
                            recurrent_regularizer=self.recurrent_regularizer)
            self.layers.append(layer)
        output_layer = Dense(units=self.plant.input_dim,
                             activation='sigmoid',
                             name='output_layer',
                             bias_initializer=tf.initializers.Constant(value=-5),
                             kernel_initializer=tf.initializers.random_normal(stddev=10 ** -3))
        self.layers.append(output_layer)
        self.built = True

    def get_config(self):
        cfg = super().get_config()
        return cfg

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, states=None, **kwargs):
        # unpack states
        old_joint_pos = states[0]
        old_muscle_state = states[2]
        old_geometry_state = states[3]
        old_proprio_feedback = states[4]
        old_visual_feedback = states[5]
        old_hidden_states = states[6:]
        new_hidden_states_dict = {}
        new_hidden_states = []

        # split perturbation signal out of the back of inputs
        # the perturbation signal must be the last 2 dimensions of inputs
        if self.perturbation_dim_start is not None:
            inputs, perturbation = tf.split(inputs, [inputs.shape[1]-self.perturbation_dim_start, self.perturbation_dim_start], axis=1)

        # take out feedback backlog
        proprio_backlog = tf.slice(old_proprio_feedback, [0, 0, 1], [-1, -1, -1])
        visual_backlog = tf.slice(old_visual_feedback, [0, 0, 1], [-1, -1, -1])

        # concatenate inputs for this timestep
        proprio_fb = tf.squeeze(tf.slice(old_proprio_feedback, [0, 0, 0], [-1, -1, 1]), axis=-1)
        visual_fb = tf.squeeze(tf.slice(old_visual_feedback, [0, 0, 0], [-1, -1, 1]), axis=-1)
        x = tf.concat((proprio_fb, visual_fb, inputs), axis=-1)

        # forward pass
        for k in range(self.n_hidden_layers):
            x, new_hidden_state = self.layers[k](x, old_hidden_states[k])
            new_hidden_states_dict['gru_hidden' + str(k)] = new_hidden_state
            new_hidden_states.append(new_hidden_state)
        u = self.layers[-1](x)
        if self.perturbation_dim_start is not None:
            jstate, cstate, mstate, gstate = self.plant(u, old_joint_pos, old_muscle_state, old_geometry_state,
                                                        joint_load=perturbation)
        else:
            jstate, cstate, mstate, gstate = self.plant(u, old_joint_pos, old_muscle_state, old_geometry_state)

        # add feedback noise & update feedback backlog
        muscle_len = tf.slice(mstate, [0, 1, 0], [-1, 1, -1]) / self.plant.Muscle.l0_ce
        muscle_vel = tf.slice(mstate, [0, 2, 0], [-1, 1, -1]) / self.plant.Muscle.vmax
        # flatten len / vel / n_muscles:
        proprio_true = tf.reshape(tf.concat([muscle_len, muscle_vel], axis=1), shape=(-1, self.n_muscles * 2))
        visual_true, _ = tf.split(cstate, 2, axis=-1)  # position only (discard velocity)
        proprio_noisy = proprio_true + tf.random.normal(tf.shape(proprio_true), stddev=self.proprioceptive_noise_sd)
        visual_noisy = visual_true + tf.random.normal(tf.shape(visual_true), stddev=self.visual_noise_sd)
        new_proprio_feedback = tf.concat([proprio_backlog, proprio_noisy[:, :, tf.newaxis]], axis=2)
        new_visual_feedback = tf.concat([visual_backlog, visual_noisy[:, :, tf.newaxis]], axis=2)

        # pack new states
        new_states = [jstate, cstate, mstate, gstate, new_proprio_feedback, new_visual_feedback]
        new_states.extend(new_hidden_states)

        # pack output
        output = {'joint position': jstate,
                  'cartesian position': cstate,
                  'muscle state': mstate,
                  'geometry state': gstate,
                  'proprioceptive feedback': new_proprio_feedback,
                  'visual feedback': new_visual_feedback,
                  **new_hidden_states_dict}

        return output, new_states

    def get_initial_state(self, inputs=None, batch_size=1, dtype=tf.float32):
        if inputs is not None:
            states = self.plant.get_initial_state(joint_state=inputs, batch_size=batch_size)
        else:
            states = self.plant.get_initial_state(batch_size=batch_size)
        hidden_states = [tf.zeros((batch_size, n_units), dtype=dtype) for n_units in self.n_units]

        # flatten len / vel / n_muscles for proprioception feedback
        proprio_true = tf.reshape(states[2][:, 1:3, :], shape=(batch_size, self.n_muscles * 2))
        visual_true, _ = tf.split(states[1], 2, axis=-1)  # position only (discard velocity)
        proprio_feedback = tf.tile(proprio_true[:, :, tf.newaxis], [1, 1, self.proprioceptive_delay])
        visual_feedback = tf.tile(visual_true[:, :, tf.newaxis], [1, 1, self.visual_delay])

        # add feedback noise
        proprio_feedback += tf.random.normal(tf.shape(proprio_feedback), stddev=self.proprioceptive_noise_sd)
        visual_feedback += tf.random.normal(tf.shape(visual_feedback), stddev=self.visual_noise_sd)

        states.append(proprio_feedback)
        states.append(visual_feedback)
        states.extend(hidden_states)
        return states
