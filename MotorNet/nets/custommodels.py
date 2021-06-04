import tensorflow as tf
from tensorflow import keras
import json
import os


class MotorNetModel(keras.Model):
    def __init__(self, inputs, outputs, name='controller', **kwargs):
        self.inputs = inputs
        self.outputs = outputs
        self.task = kwargs.get('task', None)
        super(MotorNetModel, self).__init__(inputs=inputs, outputs=outputs, name=name)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # we can recompute y after the forward pass if it's required by the task
            if self.task.do_recompute_targets:
                y = self.task.recompute_targets(x, y, y_pred)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def save_model(self, path, **kwargs):
        cfg = {'Task': self.task.get_save_config()}
        cfg.update({'Controller': self.task.controller.get_save_config()})
        cfg.update({'Plant': self.task.plant.get_save_config()})
        loss_history = kwargs.get('loss_history', None)
        cfg.update({'Training Log': loss_history})
        if os.path.isfile(path + '.json') and loss_history is None:
            raise ValueError('Configuration file already exists')
        else:
            file = open(path + '.json', 'w+')
            json.dump(cfg, file)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'task': self.task, 'inputs': self.inputs, 'outputs': self.outputs})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)