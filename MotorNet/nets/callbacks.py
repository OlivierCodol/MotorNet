import copy
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend


class BatchLogger(Callback):
    """A Logger that logs metrics per batch steps instead of per epoch step."""
    def __init__(self):
        super().__init__()
        self.history = {}
        self.weights_log = []

    def on_train_begin(self, logs=None):
        # log initial weights
        self.weights_log.append(copy.deepcopy(self.model.weights))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for metric, v in logs.items():
            self.history.setdefault(metric, []).append(v)
        self.weights_log.append(copy.deepcopy(self.model.weights))


class TrainingPlotter(Callback):
    def __init__(self, task, plot_freq=20):
        super().__init__()
        self.task = task
        self.plot_freq = plot_freq
        self.logs = []
        self.losses = {}
        self.loss_weights = {task.loss_names[loss] + '_loss': weight for loss, weight in self.task.loss_weights.items()}

    def on_train_begin(self, logs=None):
        self.logs = []
        self.losses = {}

    def on_batch_end(self, batch, logs=None):

        if self.losses == {}:  # initialize the losses dictionary if needed
            self.losses = {name.replace('_', ' '): [] for name in logs.keys() if name.__contains__('loss')}
            self.losses['kernel weight loss'] = []
            self.losses['recurrent weight loss'] = []
            self.losses['output kernel loss'] = []

        for loss, val in logs.items():
            if loss.__contains__('loss'):
                if loss.__contains__('_loss'):
                    weight = self.loss_weights[loss]
                else:
                    weight = 1
                self.losses[loss.replace('_', ' ')].append(np.array(val) * weight)

        self.losses['kernel weight loss'].append(np.array(self.model.losses[0]))
        self.losses['recurrent weight loss'].append(np.array(self.model.losses[1]))
        self.losses['output kernel loss'].append(np.array(self.model.losses[2]))

        self.logs.append(logs)

        if batch % self.plot_freq == 0 or batch == 1:
            [inputs, targets, init_states] = self.task.generate(batch_size=3,
                                                                n_timesteps=self.task.training_n_timesteps)
            results = self.model([inputs, init_states], training=False)

            if self.task.do_recompute_targets:
                targets = self.task.recompute_targets((inputs, init_states), targets, results)

            c_results = results['cartesian position']
            m_results = results['muscle state']
            h_results = results['gru_hidden0']

            clear_output(wait=True)
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(1, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            for loss, val in self.losses.items():
                ax1.plot(val, label=loss)
            ax1.set(xlabel='iteration', ylabel='weighted loss value')
            ax1.legend()
            plt.show()

            for trial in range(targets.shape[0]):
                plt.figure(figsize=(14, 2.5)).set_tight_layout(True)

                plt.subplot(141)
                plt.plot(np.array(targets[trial, :, 0]).squeeze(), color='#1f77b4', linestyle='dashed')
                plt.plot(np.array(targets[trial, :, 1]).squeeze(), color='#ff7f0e', linestyle='dashed')
                plt.plot(np.array(c_results[trial, :, 0]).squeeze(), color='#1f77b4', label='x')
                plt.plot(np.array(c_results[trial, :, 1]).squeeze(), color='#ff7f0e', label='y')
                plt.legend()
                plt.xlabel('time (ms)')
                plt.ylabel('x/y position')

                plt.subplot(142)
                plt.plot(np.array(m_results[trial, :, 0, :]).squeeze())
                plt.xlabel('time (ms)')
                plt.ylabel('activation (a.u.)')

                plt.subplot(143)
                plt.plot(np.array(m_results[trial, :, 2, :]).squeeze())
                plt.xlabel('time (ms)')
                plt.ylabel('muscle velocity (m/sec)')

                plt.subplot(144)
                plt.plot(np.array(h_results[trial, :, :]).squeeze())
                plt.xlabel('time (ms)')
                plt.ylabel('hidden unit activity')

                plt.show()


class CustomLearningRateScheduler(LearningRateScheduler):
    def __init__(self, scheduler, verbose=0):
        super().__init__(scheduler, verbose)

    def on_epoch_begin(self, epoch, logs=None):
        return None
    
    def on_batch_end(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(backend.get_value(self.model.optimizer.lr))
            lr = self.schedule(batch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(batch)
        if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        if self.verbose > 0:
            print('\nBatch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (batch + 1, lr))


# See https://github.com/tensorflow/tensorflow/issues/42872
class TensorflowFix(Callback):
    def __init__(self):
        super(TensorflowFix, self).__init__()
        self._supports_tf_logs = True
        self._backup_loss = None

    def on_train_begin(self, logs=None):
        self._backup_loss = {**self.model.loss}

    def on_train_batch_end(self, batch, logs=None):
        self.model.loss = self._backup_loss
