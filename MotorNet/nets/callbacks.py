import copy
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

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
    def __init__(self, task):
        super().__init__()
        self.task = task

    def on_train_begin(self, logs=None):
        self.loss = []
        self.logs = []
        self.cartesian_loss = []
        self.muscle_loss = []

    def on_batch_end(self, batch, logs=None):
        self.logs.append(logs)
        self.loss.append(logs.get('loss'))
        self.cartesian_loss.append(logs.get('RNN_loss'))
        self.muscle_loss.append(logs.get('RNN_4_loss'))

        if len(self.loss) > 1 and batch % 10 == 0:
            clear_output(wait=True)
            N = np.arange(0, len(self.loss))
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(1,1)
            ax1 = fig.add_subplot(gs[0,0])
            ax1.plot(N, self.cartesian_loss, label='cartesian loss')
            ax1.plot(N, self.muscle_loss, label='activation loss')
            ax1.set(xlabel='iteration', ylabel='loss')
            ax1.legend()
            plt.show()
