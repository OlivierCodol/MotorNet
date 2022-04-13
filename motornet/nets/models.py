import tensorflow as tf
import json
import os
from abc import ABC


class MotorNetModel(tf.keras.Model, ABC):
    """This is a custom ``tensorflow.keras.Model`` object, whose purpose is to enable saving
    ``motornet.plants`` object configuration when saving the model as well.

    In Tensorflow, ``tensorflow.keras.Model`` objects group layers into an object with training and inference features.
    See the Tensorflow documentation for more details on how to declare, compile and use use a
    ``tensorflow.keras.Model`` object.

    Args:
        inputs: The input(s) of the model: a ``tensorflow.keras.layers.Input`` object or list of
            ``tensorflow.keras.layers.Input`` objects.
        outputs: The output(s) of the model. See Functional API example in the Tensorflow documentation.
        task: A ``motornet.task.Task`` object or subclass.
        name: String, the name of the model.
    """

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
        """The logic for one training step. Compared to the default method, this overriding method allows for
        recomputation of targets online (during movement), in addition to essentially reproducing what the default
        method does. Notable features missing from the original method include outputing metrics as a list instead of a
        dictionary (since `motornet` always uses dictionaries) and the use of sample weighting, since data is usually
        synthetic and not empirical, avoiding sampling bias altogether.

        Args:
            data: A nested structure of `Tensor` s.

        Returns:
            A `dict` containing values that will be passed to :meth:`tf.keras.callbacks.CallbackList.on_train_batch_end`.
            Typically, the values of the `Model`'s metrics are returned. Example: `{'loss': 0.2, 'accuracy': 0.7}`.
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
        """Gets the model's configuration as a dictionary and then save it into a JSON file.

        Args:
            path: String, the absolute path to the JSON file that will be produced. The name of the JSON file itself
                should be included, without the extension. For instance, if we want to create a JSON file called
                `my_model_config.json` in `~/path/to/desired/directory`, we would call this method in the python console
                like so:

                .. code-block:: python

                    model.save_model("~/path/to/desired/directory/my_model_config")

            **kwargs: Additional keyword arguments. They are not used here, and are only present for subclassing
                compatibility.
        """
        cfg = {'Task': self.task.get_save_config()}
        cfg.update({'Network': self.task.network.get_save_config()})
        cfg.update({'Plant': self.task.network.plant.get_save_config()})
        if os.path.isfile(path + '.json'):
            raise ValueError('Configuration file already exists')
        else:
            with open(path + '.json', 'w+') as file:
                json.dump(cfg, file)

    def get_config(self):
        """Get the model's configuration.

        Returns:
            A dictionary containing the model's configuration. This includes the task object passed at initialization.
        """

        cfg = super().get_config()
        cfg.update({'task': self.task, 'inputs': self.inputs, 'outputs': self.outputs})
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
