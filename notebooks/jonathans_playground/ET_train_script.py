import numpy as np
import tensorflow as tf
import os, sys
from tensorflow.keras.layers import Input
from multiprocessing import Pool

this_file = sys.argv[0]
save_name = sys.argv[1]
num_training_steps = int(sys.argv[2])
num_iters = int(sys.argv[3])
print(save_name, num_training_steps, num_iters)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# find root directory and add to path
root_index = os.getcwd().rfind('notebooks')
root_directory = os.path.dirname(os.getcwd()[:root_index])
sys.path.append(root_directory)

from MotorNet.plants import RigidTendonArm
from MotorNet.plants.muscles import RigidTendonHillMuscle, RigidTendonHillMuscleThelen
from MotorNet.utils.plotor import plot_pos_over_time
from MotorNet.nets.layers import GRUController
from MotorNet.nets.callbacks import BatchLogger, TrainingPlotter, CustomLearningRateScheduler, TensorflowFix
from MotorNet.tasks.tasks import TaskLoadProbability, TaskStaticTarget, TaskDelayedReach, TaskStaticTargetWithPerturbations, TaskDelayedMultiReach
from MotorNet.nets.custommodels import MotorNetModel


def f(run_iter):
    # tf.debugging.enable_check_numerics()
    print('tensorflow version: ' + tf.__version__)

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
    inputs = Input((None, task.get_input_dim()))
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


    #name = '50gru_1e-3dt_weights'
    #control_rnn.load_weights(os.getcwd() + '/saved_models/' + name)
    #task = control_rnn.task

    batch_size = 32
    n_t = int(2.0 / arm.dt)
    task.training_mode = True
    task.set_training_params(batch_size=batch_size, n_timesteps=n_t, iterations=num_training_steps)
    with tf.device('/cpu:0'):
        control_rnn.fit(task, verbose=1,
                        callbacks=[tensorflowfix_callback],
                        shuffle=False)

    ### save model
    name = save_name + '_' + str(run_iter)
    control_rnn.save_weights('/home/jonathan/Desktop/MotorNetModels/LoadProb/Nets/' + name)


if __name__ == '__main__':
    p = Pool(num_iters)
    p.map(f, range(num_iters))



