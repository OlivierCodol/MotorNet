import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from abc import abstractmethod
from motornet.nets.losses import PositionLoss, L2xDxActivationLoss, L2xDxRegularizer, CompoundedLoss
from copy import deepcopy
from typing import Union


class Task(tf.keras.utils.Sequence):
    """Base class for tasks.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        initial_joint_state: `Tensor` or `numpy.ndarray`, the desired initial joint states for the task, if a single set
            of pre-defined initial joint states is desired. If `None`, the initial joint states will be drawn from the
            :class:`motornet.nets.layers.Network.get_initial_state` method at each call of :meth:`generate`.

            This parameter will be ignored on :meth:`generate` calls where a `joint_state` is provided as input
            argument.
        name: `String`, the name of the task object instance.
    """
    def __init__(self, network, initial_joint_state=None, name: str = 'Task'):
        self.__name__ = name
        self.network = network
        self.dt = self.network.plant.dt
        # self.training_iterations = 1000
        self.training_batch_size = 32
        self.training_n_timesteps = 100
        self.delay_range = [0, 0]
        self.do_recompute_targets = False
        self.losses = {name: None for name in self.network.output_names}
        self.loss_names = {name: name for name in self.network.output_names}
        self.loss_weights = {name: 0. for name in self.network.output_names}
        self._losses = {name: [] for name in self.network.output_names}
        self._loss_names = {name: [] for name in self.network.output_names}
        self._loss_weights = {name: [] for name in self.network.output_names}

        if initial_joint_state is not None:
            initial_joint_state = np.array(initial_joint_state)
            if len(initial_joint_state.shape) == 1:
                initial_joint_state = initial_joint_state.reshape(1, -1)
            self.n_initial_joint_states = initial_joint_state.shape[0]
            self.initial_joint_state_original = initial_joint_state.tolist()
        else:
            self.initial_joint_state_original = None
            self.n_initial_joint_states = None
        self.initial_joint_state = initial_joint_state

        self.convert_to_tensor = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(x))

    def add_loss(self, assigned_output, loss, loss_weight=1., name=None):
        """Add a loss to optimize during training.

        Args:
            assigned_output: `String`, the output state that the loss will be applied to. This should correspond to
                an output name from the :class:`Network` object instance passed at initialization. The output names
                can be retrieved via the :attr:`motornet.nets.layers.Network.output_names` attribute.
            loss: :class:`tensorflow.python.keras.losses.Loss` object class or subclass. `Loss`
                subclasses specific to `MotorNet` are available in the :class:`motornet.nets.losses` module.
            loss_weight: `Float`, the weight of the loss when all contributing losses are added to the total loss.
            name: `String`, the name (label) to give to the loss object. This is used to print, plot, and
                save losses during training.

        Raises:
            ValueError: If the `assigned_output` passed does not match any network output name.
        """
        if assigned_output not in self.network.output_names:
            raise ValueError("The assigned output passed does not match any network output name.")

        is_compounded = True if self._losses[assigned_output] else False
        keep_default_name = False

        if name is not None:
            # if a name is given, overwrite the default name assigned at initialization
            self._loss_names[assigned_output].append(name)
        elif hasattr(loss, 'name'):
            # else if the loss object has a name attribute, then use that name instead
            self._loss_names[assigned_output].append(loss.name)
        else:
            keep_default_name = True
            self._loss_names[assigned_output].append('subloss_' + str(len(self._loss_names[assigned_output] + 1)))

        if is_compounded:
            self.loss_names[assigned_output] = assigned_output.replace(' ', '_') + '_compounded'
        elif keep_default_name is False:
            self.loss_names[assigned_output] = self._loss_names[assigned_output][0]

        self.loss_weights[assigned_output] = 1. if is_compounded else loss_weight
        self._loss_weights[assigned_output].append(loss_weight)

        self._losses[assigned_output].append(deepcopy(loss))
        if is_compounded:
            losses = self._losses[assigned_output]
            loss_weights = self._loss_weights[assigned_output]
            self.losses[assigned_output] = CompoundedLoss(losses=losses, loss_weights=loss_weights)
        else:
            self.losses[assigned_output] = self._losses[assigned_output][0]

    @abstractmethod
    def generate(self, batch_size, n_timesteps, validation: bool = False):
        """Generates inputs, targets, and initial states to be passed to the `model.fit` call.

        Args:
            batch_size: `Integer`, the batch size to use to create the inputs, targets, and initial states.
            n_timesteps: `Integer`, the number of timesteps to use to create the inputs and targets. Initial states do
                not require a time dimension.
            validation: `Boolean`, whether to generate trials for validation purposes or not (as opposed to training
                purposes). This is useful when one wants to test a network's performance in a set of trial types that
                are not the same as those used for training.

        Returns:
            - A `dictionary` to use as input to the model. Each value in the `dictionary` should be a `tensor` array.
              At the very least, the `dictionary` should contain a "inputs" key mapped to a `tensor` array, which will
              be passed as-is to the network's input layer. Additional keys will be passed and handled didderently
              depending on what the :class:`Network` passed at initialization does when it is called.
            - A `tensor` array of target values, that will be passed to all losses as the `y_true` input to compute
              loss values.
            - A `list` of initial state as `tensor` arrays, compatible with the :attr:`initial_joint_state` value set at
              initialization.
        """
        return

    def get_initial_state(self, batch_size, joint_state=None):
        """Computes initial state instances that are biomechanically compatible with each other.

        Args:
            batch_size: `Integer`, the batch size defining the size of each state's first dimension.
            joint_state: The joint state from which the other state values are inferred. If provided, this is passed
                as-is to the :meth:`motornet.nets.layers.Network.get_initial_state` method, and therefore obeys the
                logic documented there. If not provided and an :attr:`initial_joint_state` was defined at
                initialization, that:attr:`initial_joint_state` will be passed instead. If neither is provided, then
                `None` is passed as "input" argument.

        Returns:
            A 'list' containing the output of the :meth:`motornet.nets.layers.Network.get_initial_state` method,
            which is usually a `tensor` array per state.
        """
        if joint_state is None:
            if self.initial_joint_state is None:
                inputs = None
            else:
                i = np.random.randint(0, self.n_initial_joint_states, batch_size)
                inputs = self.initial_joint_state[i, :]
            initial_states = self.network.get_initial_state(batch_size=batch_size, inputs=inputs)
        else:
            initial_states = self.network.get_initial_state(batch_size=batch_size, inputs=joint_state)
        return initial_states

    def get_input_dim(self):
        """Gets the dimensionality of each value in the input `dictionary` produced by the :meth:`generate` method.

        Returns:
            A `dictionary` with keys corresponding to those of the input `dictionary` produced by the :meth:`generate`
            method, mapped to `lists` indicating the dimensionality (shape) of each value in the input `dictionary`.
        """

        [inputs, _, _] = self.generate(batch_size=1, n_timesteps=self.delay_range[-1]+1)

        def sort_shape(i):
            if tf.is_tensor(i):
                s = i.get_shape().as_list()
            elif isinstance(i, np.ndarray):
                s = i.shape
            else:
                raise TypeError("Can only take a tensor or numpy.ndarray as input.")
            return s[-1]

        if type(inputs) is dict:
            shape = {key: sort_shape(val) for key, val in inputs.items()}
        else:
            shape = inputs

        return shape

    def get_losses(self):
        """Gets the currently declared losses and their corresponding loss weight.

        Returns:
            - A `dictionary` containing loss objects.
            - A `dictionary` containing `float` values corresponding to each loss' weight.
        """
        return [self.losses, self.loss_weights]

    def print_losses(self):
        """Prints all currently declared losses in a readable format, including the default losses declared at
        initialization. This method prints the assigned output, loss object instance, loss weight and loss name of each
        loss. It also specifies if each loss is part of a compounded loss or not.
        """
        for key, val in self._losses.items():
            if val:
                for n, elem in enumerate(val):
                    title = "ASSIGNED OUTPUT: " + key
                    print(title)
                    print("-" * len(title))
                    print("loss function: ", elem)
                    print("loss weight:   ", self._loss_weights[key][n])
                    print("loss name:     ", self._loss_names[key][n])
                    if len(val) > 1:
                        print("Compounded:     YES")
                    else:
                        print("Compounded:     NO")
                    print("\n")

    def get_attributes(self):
        """Gets all non-callable attributes declared in the object instance, except for loss-related attributes.

        Returns:
            - A `list` of attribute names as `string` elements.
            - A `list` of attribute values.
        """
        blacklist = ['loss_weights', 'losses', 'loss_names']
        attributes = [
            a for a in dir(self)
            if not a.startswith('_') and not callable(getattr(self, a)) and not blacklist.__contains__(a)
        ]
        values = [getattr(self, a) for a in attributes]
        return attributes, values

    def print_attributes(self):
        """Prints all non-callable attributes declared in the object instance, except for loss-related attributes.
        To print loss-related attributes, see :meth:`print_losses`."""
        attributes = [a for a in dir(self) if not a.startswith('_') and not callable(getattr(self, a))]
        blacklist = ['loss_weights', 'losses', 'loss_names']

        for a in attributes:
            if not blacklist.__contains__(a):
                print(a + ": ", getattr(self, a))

        for elem in blacklist:
            print("\n" + elem + ":\n", getattr(self, elem))

    def set_training_params(self, batch_size, n_timesteps):
        """Sets default training parameters for the :meth:`generate` call. These will be overridden if the
        :meth:`generate` method is called with alternative values for these parameters.

        Args:
            batch_size: `Integer`, the batch size to use to create the inputs, targets, and initial states.
            n_timesteps: `Integer`, the number of timesteps to use to create the inputs and targets. Initial states do
                not require a time dimension.
        """
        self.training_batch_size = batch_size
        self.training_n_timesteps = n_timesteps
        # self.training_iterations = iterations

    def get_save_config(self):
        """Gets the task object's configuration as a `dictionary`.

        Returns:
            A `dictionary` containing the  parameters of the task's configuration. All parameters held as non-callbale
            attributes by the object instance will be included in the `dictionary`.
        """

        cfg = {'name': self.__name__}  # 'training_iterations': self.training_iterations,
        attributes, values = self.get_attributes()
        for attribute, value in zip(attributes, values):
            if isinstance(value, np.ndarray):
                print("WARNING: One of the attributes of the Task object whose configuration dictionary is being "
                      "fetched is a numpy array, which is not JSON serializable. This may result in an error when "
                      "trying to save the model containing this Task as a JSON file. This is likely to occur with a "
                      "custom Task subclass that includes a custom attribute saved as a numpy array. To avoid this, it "
                      "is recommended to ensure none of the attributes of the Task are numpy arrays.")
            cfg[attribute] = value

        # save all losses as a list of dictionaries, each containing the information for one contributing loss.
        losses = []
        for key, val in self._losses.items():
            if val:
                for n, elem in enumerate(val):
                    d = {"assigned output": key, "loss object": str(elem), "loss name": self._loss_names[key][n],
                         "loss weight": self._loss_weights[key][n]}
                    if len(val) > 1:
                        d["compounded"] = True
                    else:
                        d["compounded"] = False
                    losses.append(d)
        cfg["losses"] = losses

        return cfg

    def __getitem__(self, idx):
        [inputs, targets, init_states] = self.generate(
            batch_size=self.training_batch_size,
            n_timesteps=self.training_n_timesteps
        )
        return [inputs, init_states], targets

    # def __len__(self):
    #     return self.training_iterations

    def get_input_dict_layers(self):
        """Creates :class:`tensorflow.keras.layers.Input` layers to build the entrypoint layers of the network inputs.
        See the `tensorflow` documentation for more information about what :class:`tensorflow.keras.layers.Input`
        objects do. Below is an example code using the current method to create a model. See
        :meth:`get_initial_state_layers` for more information about how to create a set of input layers for initial
        states.

        .. code-block:: python

            import motornet as mn
            import tensorflow as tf

            plant = mn.plants.ReluPointMass24()
            network = mn.nets.layers.GRUNetwork(plant=plant, n_units=50)
            task = mn.tasks.CentreOutReach(network=network)

            rnn = tf.keras.layers.RNN(cell=network, return_sequences=True)

            inputs = task.get_input_dict_layers()
            state_i = task.get_initial_state_layers()
            state_f = rnn(inputs, initial_state=state_i)

        Returns:
            A `dictionary`, with the same keys as the ``inputs`` dictionary from the :meth:`generate` method. These keys
            are mapped onto :class:`tensorflow.keras.layers.Input` object instances with dimensionality corresponding to
            the inputs provided in the ``inputs`` dictionary from the :meth:`generate` method.
        """
        return {key: Input((None, val,), name=key) for key, val in self.get_input_dim().items()}

    def get_initial_state_layers(self):
        """Creates :class:`tensorflow.keras.layers.Input` layers to build the entrypoint layers of the network states.
        See the `tensorflow` documentation for more information about what :class:`tensorflow.keras.layers.Input`
        objects do. Below is an example code using the current method to create a model. See
        :meth:`get_input_dict_layers` for more information about how to create a set of input layers for network inputs.

        .. code-block:: python

            import motornet as mn
            import tensorflow as tf

            plant = mn.plants.ReluPointMass24()
            network = mn.nets.layers.GRUNetwork(plant=plant, n_units=50)
            task = mn.tasks.CentreOutReach(network=network)

            rnn = tf.keras.layers.RNN(cell=network, return_sequences=True)

            inputs = task.get_input_dict_layers()
            state_i = task.get_initial_state_layers()
            state_f = rnn(inputs, initial_state=state_i)

        Returns:
            A `list` of :class:`tensorflow.keras.layers.Input` object instances with dimensionality corresponding to
            the (initial) states provided by the :class:`motornet.nets.layers.Network` object instance passed at
            initialization.
        """
        shapes = self.network.state_size
        names = self.network.initial_state_names
        return [Input(shape, name=name) for shape, name in zip(shapes, names)]


class RandomTargetReach(Task):
    """A reach to a random target from a random starting position.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
            loss.
        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(self, network, name: str = 'RandomTargetReach', deriv_weight: float = 0., **kwargs):
        super().__init__(network, name=name, **kwargs)
        max_iso_force = self.network.plant.muscle.max_iso_force
        dt = self.network.plant.dt
        muscle_loss = L2xDxActivationLoss(max_iso_force=max_iso_force, dt=dt, deriv_weight=deriv_weight)
        gru_loss = L2xDxRegularizer(deriv_weight=0.05, dt=dt)
        self.add_loss('gru_hidden_0', loss_weight=0.1, loss=gru_loss)
        self.add_loss('muscle state', loss_weight=5, loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=1., loss=PositionLoss())

    def generate(self, batch_size, n_timesteps, validation: bool = False):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
        goal_states = self.network.plant.joint2cartesian(goal_states_j)
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs = {"inputs": targets[:, :, :self.network.plant.space_dim]}
        return [inputs, targets, init_states]


class RandomTargetReachWithLoads(Task):
    """A reach to a random target from a random starting position, with loads applied at the skeleton's endpoint.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        endpoint_load: `Float`, or `K`-items `list`, `tuple` or `numpy.ndarray`, with `K` the :attr:`space_dim`
            attribute of the :class:`motornet.plants.skeletons.Skeleton` object class or subclass, `i.e.`, the
            dimensionality of the worldspace. Each element `k` in `K` indicates the the load (N) to apply to the
            skeleton's endpoint for the `k`-th worldspace dimension. If a `float` is given, that load is applied to
            all dimensions.
        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
            loss.
        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(
            self,
            network,
            endpoint_load: Union[float, list, tuple, np.ndarray],
            name: str = 'RandomTargetReachWithLoads',
            deriv_weight: float = 0.,
            **kwargs
    ):

        super().__init__(network, name=name, **kwargs)
        max_iso_force = self.network.plant.muscle.max_iso_force
        dt = self.network.plant.dt
        muscle_loss = L2xDxActivationLoss(max_iso_force=max_iso_force, dt=dt, deriv_weight=deriv_weight)
        gru_loss = L2xDxRegularizer(deriv_weight=0.05, dt=dt)
        self.add_loss('gru_hidden_0', loss_weight=0.1, loss=gru_loss)
        self.add_loss('muscle state', loss_weight=5, loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=1., loss=PositionLoss())
        self.endpoint_load = endpoint_load

    def generate(self, batch_size, n_timesteps, validation: bool = False):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
        goal_states = self.network.plant.joint2cartesian(goal_states_j)
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        endpoint_load = tf.constant(self.endpoint_load, shape=(batch_size, n_timesteps, 2))
        inputs = {"inputs": targets[:, :, :self.network.plant.space_dim], "endpoint_load": endpoint_load}
        return [inputs, targets, init_states]


class DelayedReach(Task):
    """A random-delay reach to a random target from a random starting position.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
            loss.
        delay_range: Two-items `list`, `tuple` or `numpy.ndarray`, indicating the lower and upper range of the time
            window (sec) in which the go cue may be presented. The delay is randomly drawn from a uniform distribution
            bounded by these values.
        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """
    def __init__(
            self,
            network,
            name: str = 'DelayedReach',
            delay_range: Union[list, tuple] = (0.3, 0.6),
            deriv_weight: float = 0.,
            **kwargs
    ):

        super().__init__(network, name=name, **kwargs)
        max_iso_force = self.network.plant.muscle.max_iso_force
        dt = self.network.plant.dt
        muscle_loss = L2xDxActivationLoss(max_iso_force=max_iso_force, dt=dt, deriv_weight=deriv_weight)
        gru_loss = L2xDxRegularizer(deriv_weight=0.05, dt=self.network.plant.dt)
        self.add_loss('gru_hidden_0', loss_weight=0.1, loss=gru_loss)
        self.add_loss('muscle state', loss_weight=5, loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=1., loss=PositionLoss())
        delay_range = np.array(delay_range) / self.network.plant.dt
        self.delay_range = [int(delay_range[0]), int(delay_range[1])]
        self.convert_to_tensor = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(x))

    def generate(self, batch_size, n_timesteps, validation: bool = False):
        goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
        goal_states = self.network.plant.joint2cartesian(goal_states_j)
        init_states = self.get_initial_state(batch_size=batch_size)
        center = self.network.plant.joint2cartesian(init_states[0][:, :])
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()

        inputs = copy.deepcopy(targets)
        gocue = np.zeros([batch_size, n_timesteps, 1])
        for i in range(batch_size):
            delay_time = int(np.random.uniform(self.delay_range[0], self.delay_range[1]))
            targets[i, :delay_time, :] = center[i, np.newaxis, :]
            gocue[i, delay_time, 0] = 1.

        inputs = {"inputs": np.concatenate([inputs, gocue], axis=-1)}
        return [inputs, self.convert_to_tensor(targets), init_states]


class CentreOutReach(Task):
    """During training, the network will perform random reaches. During validation, the network will perform
    centre-out reaching movements.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        name: `String`, the name of the task object instance.
        angular_step: `Float`, the angular distance (deg) between each centre-out reach during validation. For instance,
            if this is `45`, the `Task` object will ask the network to perform reaches in `8` directions equally
            distributed around the center position.
        catch_trial_perc: `Float`, the percentage of catch trials during training. A catch trial is a trial where no
            go-cue occurs, ensuring the network has to learn to wait for the go cue to actually occur without trying
            to "anticipate" the timing of the go-cue.
        reaching_distance: `Float`, the reaching distance (m) for each centre-out reach during validation.
        start_position: `List`, `tuple` or `numpy.ndarray`, indicating the start position around which the centre-out
            reaches will occur during validation. There should be as many elements as degrees of freedom in the plant.
            If `None`, the start position will be defined as the center of the joint space, based on the joint limits of
            the plant.
        deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
            loss.
        go_cue_range: Two-items `list`, `tuple` or `numpy.ndarray`, indicating the lower and upper range of the time
            window (sec) in which the go cue may be presented. The go cue timing is randomly drawn from a uniform
            distribution bounded by these values.
        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(
            self,
            network,
            name: str = 'CentreOutReach',
            angular_step: float = 15,
            catch_trial_perc: float = 50,
            reaching_distance: float = 0.1,
            start_position: Union[list, tuple, np.ndarray] = None,
            deriv_weight: float = 0.,
            go_cue_range: Union[list, tuple, np.ndarray] = (0.05, 0.25),
            **kwargs
    ):

        super().__init__(network, name=name, **kwargs)

        self.angular_step = angular_step
        self.catch_trial_perc = catch_trial_perc
        self.reaching_distance = reaching_distance
        self.start_position = start_position
        if not self.start_position:
            # start at the center of the workspace
            lb = np.array(self.network.plant.pos_lower_bound)
            ub = np.array(self.network.plant.pos_upper_bound)
            self.start_position = lb + (ub - lb) / 2
        self.start_position = np.array(self.start_position).reshape(-1).tolist()

        muscle_loss = L2xDxActivationLoss(
            max_iso_force=self.network.plant.muscle.max_iso_force,
            dt=self.network.plant.dt,
            deriv_weight=deriv_weight
        )
        gru_loss = L2xDxRegularizer(deriv_weight=0.05, dt=self.network.plant.dt)
        self.add_loss('gru_hidden_0', loss_weight=0.1, loss=gru_loss)
        self.add_loss('muscle state', loss_weight=5, loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=1., loss=PositionLoss())

        go_cue_range = np.array(go_cue_range) / self.network.plant.dt
        self.go_cue_range = [int(go_cue_range[0]), int(go_cue_range[1])]
        self.delay_range = self.go_cue_range

    def generate(self, batch_size, n_timesteps, validation: bool = False):
        catch_trial = np.zeros(batch_size, dtype='float32')
        if not validation:
            init_states = self.get_initial_state(batch_size=batch_size)
            goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
            goal_states = self.network.plant.joint2cartesian(goal_states_j)
            p = int(np.floor(batch_size * self.catch_trial_perc / 100))
            catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
        else:
            angle_set = np.deg2rad(np.arange(0, 360, self.angular_step))
            reps = int(np.ceil(batch_size / len(angle_set)))
            angle = np.tile(angle_set, reps=reps)
            batch_size = reps * len(angle_set)
            catch_trial = np.zeros(batch_size, dtype='float32')

            start_jpv = np.concatenate([self.start_position, np.zeros_like(self.start_position)])[np.newaxis, :]
            start_cpv = self.network.plant.joint2cartesian(start_jpv)
            end_cp = self.reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)

        center = self.network.plant.joint2cartesian(init_states[0][:, :])
        go_cue = np.ones([batch_size, n_timesteps, 1])
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs_targ = copy.deepcopy(targets[:, :, :self.network.plant.space_dim])
        tmp = np.repeat(center[:, np.newaxis, :self.network.plant.space_dim], n_timesteps, axis=1)
        inputs_start = copy.deepcopy(tmp)
        for i in range(batch_size):
            if not validation:
                go_cue_time = int(np.random.uniform(self.go_cue_range[0], self.go_cue_range[1]))
            else:
                go_cue_time = int(self.go_cue_range[0] + np.diff(self.go_cue_range) / 2)

            if catch_trial[i] > 0.:
                targets[i, :, :] = center[i, np.newaxis, :]
            else:
                targets[i, :go_cue_time, :] = center[i, np.newaxis, :]
                inputs_start[i, go_cue_time + self.network.visual_delay:, :] = 0.
                go_cue[i, go_cue_time + self.network.visual_delay:, 0] = 0.

        return [
            {"inputs": np.concatenate([inputs_start, inputs_targ, go_cue], axis=-1)},
            self.convert_to_tensor(targets), init_states
        ]
