import sys
from multiprocessing import Pool

this_file = sys.argv[0]
save_name = sys.argv[1]
batch_size = int(sys.argv[2])
num_iters = int(sys.argv[3])
print(save_name, batch_size, num_iters)

def f(run_iter):
    import numpy as np
    import tensorflow as tf
    import os
    from tensorflow.keras.layers import Input
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # find root directory and add to path
    root_index = os.getcwd().rfind('notebooks')
    root_directory = os.path.dirname(os.getcwd()[:root_index])
    sys.path.append(root_directory)

    from MotorNet.plants import RigidTendonArm
    from MotorNet.plants.muscles import RigidTendonHillMuscleThelen
    from MotorNet.nets.layers import GRUController
    from MotorNet.nets.callbacks import BatchLogger, TrainingPlotter, CustomLearningRateScheduler, TensorflowFix
    from MotorNet.tasks.tasks import TaskLoadProbability

    from MotorNet.nets.custommodels import MotorNetModel
    # tf.debugging.enable_check_numerics()
    print('tensorflow version: ' + tf.__version__)

    #%% Create model


    #  Create model
    arm = RigidTendonArm(muscle_type=RigidTendonHillMuscleThelen(), timestep=0.01,
                         proprioceptive_delay=0.04, visual_delay=0.09,  # 0.04, 0.09 best
                         excitation_noise_sd=0.001)  # 0.0001 best
    visual_feedback_noise = 0.01  # 0.001 best
    proprio_feedback_noise = 0.001  # 0.001 best
    hidden_noise = 0.001  # 0.0001 best
    # kernel 5e-6 best
    # recurrent 1e-5 best
    cell = GRUController(plant=arm, n_units=200, kernel_regularizer=1e-6,
                         recurrent_regularizer=1e-5, name='cell',
                         proprioceptive_noise_sd=proprio_feedback_noise, visual_noise_sd=visual_feedback_noise,
                         hidden_noise_sd=hidden_noise)
    task = TaskLoadProbability(cell, initial_joint_state=np.deg2rad([60., 80., 0., 0.]),
                               delay_range=[300, 1600],  # 300, 1600 best
                               fixation_time=0,  # 0 best
                               )

    # declare inputs
    targets = Input((None, arm.state_dim,), name='target')
    inputs = Input((None, task.get_input_dim()), name='inputs')
    state0 = [Input((arm.state_dim, ), name='joint0'),
              Input((arm.state_dim, ), name='cartesian0'),
              Input((arm.muscle_state_dim, arm.n_muscles, ), name='muscle0'),
              Input((arm.geometry_state_dim, arm.n_muscles, ), name='geometry0'),
              Input((arm.n_muscles*2, arm.proprioceptive_delay, ), name='proprio_feedback0'),
              Input((arm.space_dim, arm.visual_delay, ), name='visual_feedback0')]
    state0.extend([Input((n, ), name='gru' + str(k) + '_hidden0') for k, n in enumerate(cell.n_units)])


    # wrap cell in an RNN layer
    states_out = tf.keras.layers.RNN(cell=cell, return_sequences=True, name='RNN')(inputs, initial_state=state0)
    control_rnn = MotorNetModel(inputs=[inputs, state0], outputs=states_out, name='controller', task=task)

    # pull the losses from the task itself
    [losses, loss_weights] = task.get_losses()

    # and compile
    control_rnn.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, clipnorm=1.), loss=losses,
                        loss_weights=loss_weights, run_eagerly=False)
    tensorflowfix_callback = TensorflowFix()
    control_rnn.summary()

    cell.layers[1].bias = tf.convert_to_tensor([-5.18, -6.47, -3.63, -6.42, -4.40, -6.48])

    name = save_name + '_' + str(run_iter)
    control_rnn.load_weights('/home/jonathan/Desktop/MotorNetModels/LoadProb/Nets/' + name).expect_partial()

    n_t = int(2.0 / arm.dt)


    task.training_mode=False
    task.delay_range = np.array([800, 1100]) / 1000 / task.plant.dt
    import scipy.io
    [inputs, targets, init_states] = task.generate(n_timesteps=n_t, batch_size=batch_size)

    results = control_rnn([inputs, init_states], training=False)

    j_results = results['joint position']
    c_results = results['cartesian position']
    m_results = results['muscle state']
    h_results = results['gru_hidden0']
    pro_results = results['proprioceptive feedback']
    vis_results = results['visual feedback']
    weights = control_rnn.get_weights()
    scipy.io.savemat('/home/jonathan/Desktop/MotorNetModels/LoadProb/Results/' + name + '.mat', {'joint': j_results.numpy(),
                                                                     'cartesian': c_results.numpy(),
                                                                     'muscle': m_results.numpy(),
                                                                     'inputs': inputs.numpy(),
                                                                     'targets': targets.numpy(),
                                                                     'neural': h_results.numpy(),
                                                                     'proprio': pro_results.numpy(),
                                                                     'visual': vis_results.numpy(),
                                                                     'weights': weights})

if __name__ == '__main__':
    p = Pool(num_iters)
    p.map(f, range(num_iters))



