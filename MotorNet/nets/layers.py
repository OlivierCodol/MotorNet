import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, GRUCell, Dense


class GRUController(Layer):
    def __init__(self, plant, n_units=20, n_hidden_layers=1, activation='tanh', kernel_regularizer=0.,
                 activity_regularizer=0., proprioceptive_noise_sd=0., visual_noise_sd=0., **kwargs):

        if type(n_units) == int:
            n_units = list(np.repeat(n_units, n_hidden_layers).astype('int32'))

        # set feedback noise levels
        # TODO: feedback signals should really be normalized so that noise levels are comparable between inputs
        self.proprioceptive_noise_sd = proprioceptive_noise_sd
        self.visual_noise_sd = visual_noise_sd

        # plant states
        self.proprioceptive_delay = plant.proprioceptive_delay
        self.visual_delay = plant.visual_delay
        self.state_size = [tf.TensorShape([plant.output_dim]),
                           tf.TensorShape([plant.output_dim]),
                           tf.TensorShape([plant.muscle_state_dim, plant.n_muscles]),
                           tf.TensorShape([plant.geometry_state_dim, plant.n_muscles]),
                           tf.TensorShape([plant.n_muscles*2, self.proprioceptive_delay]),
                           tf.TensorShape([2, self.visual_delay])]
        # hidden states for GRU layer(s)
        for n in n_units:
            self.state_size.append(tf.TensorShape([n]))

        self.output_size = self.state_size
        self.plant = plant
        self.kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        self.activity_regularizer = tf.keras.regularizers.l2(activity_regularizer)
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
                            activity_regularizer=self.activity_regularizer)
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
        jstate, cstate, mstate, gstate = self.plant(u, old_joint_pos, old_muscle_state, old_geometry_state)

        # add our feedback noise
        shape = tf.shape(mstate)
        mstate_temp = tf.reshape(mstate[:, 1:, :], shape=[shape[0], 2*shape[2]])
        mstate_noisy = mstate_temp + tf.random.normal(tf.shape(mstate_temp), mean=0., stddev=self.proprioceptive_noise_sd)
        cstate_noisy = cstate[:, 0:2] + tf.random.normal(tf.shape(cstate[:, 0:2]), mean=0., stddev=self.visual_noise_sd)

        # update feedback backlog
        new_proprio_feedback = tf.concat([proprio_backlog, mstate_noisy[:, :, tf.newaxis]], axis=2)
        new_visual_feedback = tf.concat([visual_backlog, cstate_noisy[:, :, tf.newaxis]], axis=2)

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

    def get_initial_state(self, inputs=None, batch_size=1, start_mode='random', dtype=tf.float32):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            states = self.plant.get_initial_state(skeleton_state=inputs, start_mode=start_mode)
        else:
            states = self.plant.get_initial_state(batch_size=batch_size, start_mode=start_mode)
        hidden_states = [tf.zeros((batch_size, n_units), dtype=dtype) for n_units in self.n_units]
        shape = tf.shape(states[2])
        temp_proprio = tf.reshape(states[2][:, 1:, :], shape=(shape[0], 2*shape[2]))
        proprio_feedback = tf.tile(temp_proprio[:, :, tf.newaxis], [1, 1, self.proprioceptive_delay])
        visual_feedback = tf.tile(states[1][:, 0:2, tf.newaxis], [1, 1, self.visual_delay])

        # add our feedback noise
        proprio_feedback += tf.random.normal(tf.shape(proprio_feedback), mean=0., stddev=self.proprioceptive_noise_sd)
        visual_feedback += tf.random.normal(tf.shape(visual_feedback), mean=0., stddev=self.visual_noise_sd)

        states.append(proprio_feedback)
        states.append(visual_feedback)
        states.extend(hidden_states)
        return states
