import numpy as np
from abc import ABC, abstractmethod
from MotorNet.nets.losses import empty_loss, position_loss, activation_squared_loss

class Task(ABC):
    @abstractmethod
    def __init__(self, plant, n_timesteps, batch_size, task_args):
        return

    @abstractmethod
    def generate(self):
        return

    @abstractmethod
    def get_input_dim(self):
        return


class TaskStaticTarget(Task):
    def __init__(self, plant, n_timesteps=1, batch_size=1, task_args={}):
        self.plant = plant
        self.task_args = task_args
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size

        self.losses = {'cartesian position': position_loss(), 'muscle state': activation_squared_loss()}
        self.loss_weights = {'cartesian position': 1, 'muscle state': 0.2}

    def generate(self, **kwargs):
        self.n_timesteps = kwargs.get('n_timesteps', self.n_timesteps)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        goal_states = self.plant.draw_random_uniform_states(batch_size=self.batch_size)
        targets = self.plant.state2target(state=self.plant.joint2cartesian(goal_states), n_timesteps=self.n_timesteps)
        return [targets, targets]

    def get_input_dim(self):
        [inputs, _] = self.generate()
        shape = inputs.get_shape().as_list()
        return shape[-1]

    def get_losses(self):
        return [self.losses, self.loss_weights]

def generate_delay_time(delaymin, delaymax, delaymode):
    if delaymode == 'random':
        delaytime = np.random.uniform(delaymin, delaymax)
    elif delaymode == 'noDelayInput':
        delaytime = 0

    return delaytime


