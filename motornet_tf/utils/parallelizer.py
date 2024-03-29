"""This python script allows training `MotorNet` models in parallel on the CPU. This is particularly useful for CPUs
with a large amount of cores, and should significantly improve training speed over many model iterations.

This script should be called from a terminal console, and takes up to four parameter inputs.

Args:
    1: `String`, directory to use. The declared directory should contain the model configurations saved as a JSON file.
        See :meth:`motornet.nets.models.DistalTeacher.save_model` for more information on how to produce a model
        configuration file.
    2: `String`, whether run this script in `train` or `test` mode.
    3: Run mode to be passed to task object. The type and content of this input will likely depend on the task
       object being used.
    4: `Integer`, total number of training iterations.

Raises:
    ValueError: If the first input's directory does not contain a model's JSON configuration file.
    ValueError: If the second input is not a `train` or `test` string.

"""

import json
import os
import sys
from joblib import Parallel, delayed


def _f(run_iter):
    # find root directory and add to path
    root_index = os.getcwd().rfind('utils')
    root_directory = os.path.dirname(os.getcwd()[:root_index])
    root_directory = os.path.dirname(root_directory[:root_index])
    sys.path.append(root_directory)

    import tensorflow as tf
    import scipy.io
    from ..plants import RigidTendonArm26
    from ..nets.layers import GRUNetwork
    from ..nets.callbacks import BatchLogger
    from ..nets.models import MotorNetModel
    print('tensorflow version: ' + tf.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    file_name = run_list[run_iter][0:-5]
    with open(run_list[run_iter], 'r') as config_file:
        cfg = json.load(config_file)

    muscle_type = cfg['Plant']['Muscle']['name']
    task_type = cfg['Task']['name']
    exec('from motornet.plants.muscles import ' + muscle_type)
    #exec('from motornet.tasks import ' + task_type)
    exec('from tasklocal import ' + task_type)

    arm = RigidTendonArm26(muscle_type=eval(muscle_type + '()'), timestep=float(cfg['Plant']['Skeleton']['dt']),
                           proprioceptive_delay=cfg['Plant']['proprioceptive_delay'] *
                                              float(cfg['Plant']['Skeleton']['dt']),
                           visual_delay=cfg['Plant']['visual_delay'] * float(cfg['Plant']['Skeleton']['dt']),
                           excitation_noise_sd=cfg['Plant']['excitation_noise_sd'])
    cell = GRUNetwork(plant=arm, n_units=cfg['Network']['n_units'],
                      kernel_regularizer=cfg['Network']['kernel_regularizer_weight'],
                      recurrent_regularizer=cfg['Network']['recurrent_regularizer_weight'],
                      proprioceptive_noise_sd=cfg['Network']['proprioceptive_noise_sd'],
                      visual_noise_sd=cfg['Network']['visual_noise_sd'],
                      hidden_noise_sd=cfg['Network']['hidden_noise_sd'],
                      activation=cfg['Network']['activation'])
    cfg['Task']['run_mode'] = task_run_mode
    task = eval(task_type +
                "(cell, **cfg['Task'])")

    rnn = tf.keras.layers.RNN(cell=cell, return_sequences=True, name='RNN')
    input_dict = task.get_input_dict_layers()
    state0 = task.get_initial_state_layers()
    states_out = rnn(input_dict, initial_state=state0)
    control_rnn = MotorNetModel(inputs=[input_dict, state0], outputs=states_out, name='model', task=task)

    # pull the losses from the task itself
    [losses, loss_weights] = task.get_losses()

    # compile
    control_rnn.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss=losses,
                        loss_weights=loss_weights, run_eagerly=False)
    batchlog_callback = BatchLogger()

    control_rnn.summary()

    if run_mode == 'train':
        # load trained weights
        if os.path.isfile(file_name + '.index'):
            control_rnn.load_weights(file_name).expect_partial()
        # train it up
        task.set_training_params(batch_size=cfg['Task']['training_batch_size'],
                                 n_timesteps=cfg['Task']['training_n_timesteps'])
        [inputs, targets, init_states] = task.generate(n_timesteps=cfg['Task']['training_n_timesteps'],
                                                       batch_size=cfg['Task']['training_batch_size'] * iters_per_batch)
        control_rnn.fit([inputs, init_states], targets, verbose=1, epochs=1,
                        batch_size=cfg['Task']['training_batch_size'],
                        callbacks=[batchlog_callback], shuffle=False)
        # save weights of the model
        control_rnn.save_weights(file_name)
        # add any info generated during the training process
        if os.path.isfile(file_name[0:-7] + '_training_log.json'):
            with open(file_name[0:-7] + '_training_log.json', 'r') as training_file:
                training_log = json.load(training_file)
            for key, value in training_log.items():
                training_log[key] = training_log[key] + batchlog_callback.history[key]
        else:
            training_log = batchlog_callback.history
        with open(file_name[0:-7] + '_training_log.json', 'w') as training_file:
            json.dump(training_log, training_file)
    elif run_mode == 'test':
        # load trained weights
        control_rnn.load_weights(file_name).expect_partial()

        # run task
        [inputs, targets, init_states] = task.generate(n_timesteps=cfg['Task']['training_n_timesteps'],
                                                       batch_size=total_training_iterations)
        results = control_rnn([inputs, init_states], training=False)

        # retrieve training history
        with open(file_name[0:-7] + '_training_log.json', 'r') as training_file:
            training_log = json.load(training_file)

        # save run as .mat file
        scipy.io.savemat(file_name[0:-7] + '.mat',
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
                          'training_log': training_log,
                          'task_config': cfg['Task']['task_kwargs'],
                          'controller_config': cfg['Network']})


if __name__ == '__main__':

    # input 1 - directory to use
    active_directory = sys.argv[1]
    # input 2 - run in train or test mode
    run_mode = str(sys.argv[2])
    if run_mode != 'train' and run_mode != 'test':
        raise ValueError('the run mode must be train or test')
    # input 3 - run mode to be passed to task object
    task_run_mode = str(sys.argv[3])
    # input 4 - total training iterations
    total_training_iterations = int(sys.argv[4])

    iters_per_batch = 500
    total_repeats = int(total_training_iterations / iters_per_batch)

    run_list = []
    for file in os.listdir(active_directory):
        if file.endswith("_config.json"):
            run_list.append(os.path.join(active_directory, file))
    if not run_list:
        raise ValueError('No configuration files found')

    iter_list = range(len(run_list))
    n_jobs = 16
    while len(iter_list) > 0:
        these_iters = iter_list[0:n_jobs]
        iter_list = iter_list[n_jobs:]
        if run_mode == 'train':
            repeats = total_repeats
        else:
            repeats = 1
        for i in range(repeats):
            print('repeat#' + str(i))
            result = Parallel(n_jobs=len(these_iters))(delayed(_f)(iteration) for iteration in these_iters)
