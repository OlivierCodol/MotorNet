import tensorflow as tf
import json
import os
from abc import ABC


class MotorNetModel(tf.keras.Model, ABC):
    def __init__(self, inputs, outputs, task, name='controller'):
        self.inputs = inputs
        self.outputs = outputs
        self.task = task
        super().__init__(inputs=inputs, outputs=outputs, name=name)

        # ensure each loss is tagged with the correct loss name, since the loss order is reshuffled in parent class
        flat_losses = tf.nest.flatten(task.losses)
        names = list(task.losses.keys())
        losses = list(task.losses.values())

        # all non-defined losses (=None) will share the output_name of the first model output with a non-defined loss
        output_names = [names[losses.index(loss)] for loss in flat_losses]
        self.output_names = output_names

        # if a defined loss object has been attributed a name, then use that name instead of the default output_name
        for k, name in enumerate(output_names):
            if hasattr(task.losses[name], 'name'):
                self.output_names[k] = task.losses[name].name

        # now we remove the names for the non-defined losses (loss=None cases)
        for k, loss in enumerate(flat_losses):
            if loss is None:
                self.output_names[k] = None

    def train_step(self, data):
        """Unpack the data. Its structure depends on your model and
        on what you pass to `fit()`.
        """

        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # we can recompute y after the forward pass if it's required by the task
            if self.task.do_recompute_targets:
                y = self.task.recompute_targets(x, y, y_pred)

            # Compute the loss value (the compiled_loss method is configured in `self.compile()`)
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
        cfg.update({'Network': self.task.network.get_save_config()})
        cfg.update({'Plant': self.task.network.plant.get_save_config()})
        if os.path.isfile(path + '.json'):
            raise ValueError('Configuration file already exists')
        else:
            with open(path + '.json', 'w+') as file:
                json.dump(cfg, file)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'task': self.task, 'inputs': self.inputs, 'outputs': self.outputs})
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
