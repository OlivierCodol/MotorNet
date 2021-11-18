import os, sys, json
from joblib import Parallel, delayed
import numpy as np

# input 1 - directory to use
active_directory = sys.argv[1]
# input 2 - run in train or test mode
run_mode = str(sys.argv[2])
if run_mode != 'train' and run_mode != 'test':
    raise ValueError('the run mode must be train or test')
# input 3 - run mode to be passed to task object
task_run_mode = str(sys.argv[3])

run_list = []
for file in os.listdir(active_directory):
    if file.endswith(".json"):
        run_list.append(os.path.join(active_directory, file))
if not run_list:
    raise ValueError('No configuration files found')

def f(run_iter):
    # find root directory and add to path
    root_index = os.getcwd().rfind('utils')
    root_directory = os.path.dirname(os.getcwd()[:root_index])
    root_directory = os.path.dirname(root_directory[:root_index])
    sys.path.append(root_directory)

    import tensorflow as tf
    import scipy.io
    from tensorflow.keras.layers import Input
    from MotorNet.plants import RigidTendonArm
    from MotorNet.nets.layers import GRUController
    from MotorNet.nets.callbacks import TensorflowFix, BatchLogger
    from MotorNet.nets.models import MotorNetModel
    print('tensorflow version: ' + tf.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    file_name = run_list[run_iter][0:-5]
    with open(run_list[run_iter], 'r') as config_file:
        cfg = json.load(config_file)

    muscle_type = cfg['Plant']['Muscle']['name']
    task_type = cfg['Task']['name']
    exec('from MotorNet.plants.muscles import ' + muscle_type)
    exec('from MotorNet.tasks import ' + task_type)

    #cfg['Plant']['excitation_noise_sd'] = 0.
    #cfg['Controller']['proprioceptive_noise_sd'] = 0.
    #cfg['Controller']['visual_noise_sd'] = 0.
    #cfg['Controller']['hidden_noise_sd'] = 0.

    arm = RigidTendonArm(muscle_type=eval(muscle_type + '()'), timestep=float(cfg['Plant']['Skeleton']['dt']),
                         proprioceptive_delay=cfg['Plant']['proprioceptive_delay'] *
                                              float(cfg['Plant']['Skeleton']['dt']),
                         visual_delay=cfg['Plant']['visual_delay'] * float(cfg['Plant']['Skeleton']['dt']),
                         excitation_noise_sd=cfg['Plant']['excitation_noise_sd'])
    cell = GRUController(plant=arm, n_units=cfg['Controller']['n_units'],
                         kernel_regularizer=cfg['Controller']['kernel_regularizer_weight'],
                         recurrent_regularizer=cfg['Controller']['recurrent_regularizer_weight'],
                         proprioceptive_noise_sd=cfg['Controller']['proprioceptive_noise_sd'],
                         visual_noise_sd=cfg['Controller']['visual_noise_sd'],
                         hidden_noise_sd=cfg['Controller']['hidden_noise_sd'],
                         activation=cfg['Controller']['activation'])
    task_kwargs = cfg['Task']['task_kwargs']
    task_kwargs['run_mode'] = task_run_mode
    task = eval(task_type +
                "(cell, initial_joint_state=cfg['Task']['initial_joint_state'],**task_kwargs)")

    # declare inputs
    inputs = {key: Input((None, val,), name=key) for key, val in task.get_input_dim().items()}
    state0 = [Input((arm.state_dim,), name='joint0'),
              Input((arm.state_dim,), name='cartesian0'),
              Input((arm.muscle_state_dim, arm.n_muscles,), name='muscle0'),
              Input((arm.geometry_state_dim, arm.n_muscles,), name='geometry0'),
              Input((arm.n_muscles * 2, arm.proprioceptive_delay,), name='proprio_feedback0'),
              Input((arm.space_dim, arm.visual_delay,), name='visual_feedback0')]
    state0.extend([Input((n,), name='gru' + str(k) + '_hidden0') for k, n in enumerate(cell.n_units)])

    # wrap cell in an RNN layer
    states_out = tf.keras.layers.RNN(cell=cell, return_sequences=True, name='RNN')(inputs, initial_state=state0)
    control_rnn = MotorNetModel(inputs=[inputs, state0], outputs=states_out, name='controller', task=task)

    # pull the losses from the task itself
    [losses, loss_weights] = task.get_losses()

    # compile
    control_rnn.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, clipnorm=1.), loss=losses,
                        loss_weights=loss_weights, run_eagerly=False)
    tensorflowfix_callback = TensorflowFix()
    batchlog_callback = BatchLogger()

    control_rnn.summary()

    ## SPECIAL
    cell.layers[1].bias = tf.convert_to_tensor([-5.18, -6.47, -3.63, -6.42, -4.40, -6.48])

    if run_mode == 'train':
        # load trained weights
        if os.path.isfile(file_name + '.index'):
            control_rnn.load_weights(file_name).expect_partial()
        # train it up
        iters_per_batch = cfg['Task']['training_iterations']
        task.set_training_params(batch_size=cfg['Task']['training_batch_size'],
                                 n_timesteps=cfg['Task']['training_n_timesteps'],
                                 iterations=iters_per_batch)
        [inputs, targets, init_states] = task.generate(n_timesteps=cfg['Task']['training_n_timesteps'],
                                                       batch_size=iters_per_batch * cfg['Task']['training_batch_size'])
        control_rnn.fit([inputs, init_states], targets, verbose=1, epochs=1,
                        batch_size=cfg['Task']['training_batch_size'],
                        callbacks=[tensorflowfix_callback, batchlog_callback], shuffle=False)
        # save weights of the model
        control_rnn.save_weights(file_name)
        # add any info generated during the training process
        control_rnn.save_model(file_name, loss_history=batchlog_callback.history)
    elif run_mode == 'test':
        # load trained weights
        control_rnn.load_weights(file_name).expect_partial()

        # run task
        [inputs, targets, init_states] = task.generate(n_timesteps=cfg['Task']['training_n_timesteps'],
                                                       batch_size=cfg['Task']['training_batch_size'])
        results = control_rnn([inputs, init_states], training=False)

        # save run as .mat file
        scipy.io.savemat(file_name + '.mat',
                         {'joint': results['joint position'].numpy(),
                          'cartesian': results['cartesian position'].numpy(),
                          'muscle': results['muscle state'].numpy(),
                          'inputs': inputs["inputs"],
                          'joint_load': inputs["joint_load"],
                          'targets': targets.numpy(),
                          'neural': results['gru_hidden0'].numpy(),
                          'proprio': results['proprioceptive feedback'].numpy(),
                          'visual': results['visual feedback'].numpy(),
                          'weights': control_rnn.get_weights(),
                          'training': cfg['Training Log'],
                          'task_config': cfg['Task']['task_kwargs'],
                          'controller_config': cfg['Controller']})


if __name__ == '__main__':
    iter_list = range(len(run_list))
    n_jobs = 16
    while len(iter_list) > 0:
        these_iters = iter_list[0:n_jobs]
        iter_list = iter_list[n_jobs:]
        if run_mode == 'train':
            repeats = 1
        else:
            repeats = 1
        for i in range(repeats):
            print('repeat#' + str(i))
            result = Parallel(n_jobs=n_jobs)(delayed(f)(iteration) for iteration in these_iters)
