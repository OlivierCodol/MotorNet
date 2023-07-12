"""This module implements custom :class:`tensorflow.python.keras.losses.Loss` objects that are useful for motor control.

.. note::
    There are a couple naming conventions that this module employs:

    - ``Regularizer`` indicates the penalization is applied to the *value* of the model's output, not the error
      between the model's output and a user-fed label.

    - ``xDx`` indicates that both the value and the derivative of the passed input is penalized, with the derivative
      loss scaled by a used-defined `deriv_weight` scalar compared to the value.
"""

import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper
from typing import Union

auto_reduction = losses_utils.ReductionV2.AUTO


class CompoundedLoss(LossFunctionWrapper):
    """Wraps several losses into a single loss object, creating a composite loss that sums the loss value of
    each subloss. Each subloss's contribution can be weighted by a constant scalar value.

    Args:
        losses: `List` or `tuple` of loss objects.
        loss_weights: `List` or `tuple` of `float` scalars indicating the weight of the
            corresponding loss in the `losses` argument provided above.
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensorflow` documentation for more details.
    """
    def __init__(self, losses: Union[list, tuple], loss_weights: Union[list, tuple], name: str = 'compounded_loss',
                 reduction=auto_reduction):
        super().__init__(_compounded_losses, name=name, reduction=reduction, losses=losses, loss_weights=loss_weights)


class L2xDxRegularizer(LossFunctionWrapper):
    """Applies a L2 penalty to the model's output values and value derivatives.
    For instance, if we label an output value ``x``, and its derivative ``dx``, then the penalty would evaluate at:

    .. code-block:: python

        loss = np.reduce_mean(x ** 2) + deriv_weight * np.reduce_mean(dx ** 2)

    Args:
        deriv_weight: `Float`, the weight of the derivative's penalty compared to the value itself.
        dt: `Float`, the size of a single timestep. This is used to calculate derivatives using Euler's method.
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensoflow` documentation for more details.
    """

    def __init__(self, deriv_weight: float, dt: float, name: str = 'gru_regularizer', reduction=auto_reduction):
        super().__init__(_l2_xdx_regularizer, name=name, reduction=reduction, deriv_weight=deriv_weight, dt=dt)


class L2Regularizer(LossFunctionWrapper):
    """Applies a L2 penalty to the model's output values.
    For instance, if we label an output value ``x``, then the penalty would evaluate at:

    .. code-block:: python

        loss = np.reduce_mean(x ** 2)

    Args:
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensoflow` documentation for more details.
    """

    def __init__(self, name: str = 'l2_regularizer', reduction=auto_reduction):
        super().__init__(_l2_regularizer, name=name, reduction=reduction)


class RecurrentActivityRegularizer(LossFunctionWrapper):
    """Applies a L2 penalty to the model's recurrent activity and hidden activity values.
    For instance, if we label the recurrent activity ``f``, and the hidden activity ``h``, then the
    penalty would evaluate at:

    .. code-block:: python

        loss = recurrent_weight * f + activity_weight * np.reduce_mean(h ** 2)

    The variable `f` was calculated according to the recurrent regularization method proposed in `[1]` but adapted for
    GRU units.

    References:
        [1] `Sussillo, D., Churchland, M., Kaufman, M. et al. A neural network that finds a naturalistic solution for
        the production of muscle activity. Nat Neurosci 18, 1025–1033 (2015). https://doi.org/10.1038/nn.4042`

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. The Network object must be the one
            being trained. This is used to fetch the weight values of the layer indexed `1`.
        recurrent_weight: `Float`, the weight of the penalization for the recurrent activity values.
        activity_weight: `Float`, the weight of the penalization for the hidden activity values.
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensorflow` documentation for more details.
    """

    def __init__(self, network, recurrent_weight: float, activity_weight: float, name: str = 'recurrent_activity',
                 reduction=auto_reduction):
        super().__init__(_recurrent_activity_loss, network=network, recurrent_weight=recurrent_weight,
                         activity_weight=activity_weight, name=name, reduction=reduction)


class PositionLoss(LossFunctionWrapper):
    """Applies a L1 penalty to positional error between the model's output positional state ``x`` and a user-fed
    label position ``y``:

    .. code-block:: python

        xp, _ = np.split(x, 2, axis=-1)  # remove velocity from the positional state
        yp, _ = np.split(y, 2, axis=-1)
        loss = np.reduce_mean(np.abs(xp - yp))

    .. note::
        The positional error does not include velocity, hence the use of ``np.split`` to extract position from the
        state array.

    Args:
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensoflow` documentation for more details.
    """

    def __init__(self, name: str = 'position', reduction=auto_reduction):
        super().__init__(_position_loss, name=name, reduction=reduction)


class L2ActivationLoss(LossFunctionWrapper):
    """Applies a L2 penalty to muscle activation. Must be applied to the `muscle state` output state.
    The L2 penalty is normalized by the maximum isometric force of each muscle.

    Args:
        max_iso_force: `Float` or `list`, the maximum isometric force of each muscle in the order they are declared in
            the :class:`motornet.plants.plants.Plant` object class or subclass.
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensorflow` documentation for more details.
    """

    def __init__(self, max_iso_force, name: str = 'l2_activation', reduction=auto_reduction):
        super().__init__(_l2_activation_loss, name=name, reduction=reduction, max_iso_force=max_iso_force)
        self.max_iso_force = max_iso_force


class L2ActivationMuscleVelLoss(LossFunctionWrapper):
    """Applies a L2 penalty to muscle activation and muscle velocity.
    Must be applied to the `muscle state` output state.
    The L2 penalty on muscle activation is normalized by the maximum isometric force of each muscle.
    If ``a`` is the normalized muscle activation and ``dx`` the muscle velocity, then the penalty would evaluate at:

    .. code-block:: python

        loss = tf.reduce_mean(a ** 2) + deriv_weight * tf.reduce_mean(dx ** 2)

    Args:
        max_iso_force: `Float` or `list`, the maximum isometric force of each muscle in the order they are declared in
            the :class:`motornet.plants.plants.Plant` object class or subclass.
        deriv_weight: `Float`, the weight of the derivative's penalty compared to the value itself.
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensoflow` documentation for more details.
    """

    def __init__(self, max_iso_force: float, deriv_weight: float, name: str = 'l2_activation_muscle_vel',
                 reduction=auto_reduction):
        super().__init__(_l2_activation_muscle_vel_loss, name=name, reduction=reduction, max_iso_force=max_iso_force,
                         deriv_weight=deriv_weight)


class L2ActivationL1MuscleVelIndLoss(LossFunctionWrapper):
    """Applies a L2 penalty to muscle activation and an L1 penalty to muscle velocity.
    Must be applied to the `muscle state` output state.
    The L2 penalty on muscle activation is normalized by the maximum isometric force of each muscle.
    If ``a`` is the normalized muscle activation and ``dx`` the muscle velocity, then the penalty would evaluate at:

    .. code-block:: python

        loss = activation_weight * tf.reduce_mean(a ** 2) + deriv_weight * tf.reduce_mean(tf.abs(dx))

    Args:
        max_iso_force: `Float` or `list`, the maximum isometric force of each muscle in the order they are declared in
            the :class:`motornet.plants.plants.Plant` object class or subclass.
        activation_weight: `Float`, the weight of the activation's penalty compared to the value itself.
        deriv_weight: `Float`, the weight of the derivative's penalty compared to the value itself.
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensoflow` documentation for more details.
    """

    def __init__(self, max_iso_force: float, activation_weight: float, deriv_weight: float,
                 name: str = 'l2_activation_muscle_vel', reduction=auto_reduction):
        super().__init__(_l2_activation_l1_muscle_vel_ind_loss, name=name, reduction=reduction,
                         max_iso_force=max_iso_force, activation_weight=activation_weight, deriv_weight=deriv_weight)


class L2xDxActivationLoss(LossFunctionWrapper):
    """Applies a L2 penalty to muscle activation and its derivative. Must be applied to the ``muscle state`` output
    state. The L2 penalty is normalized by the maximum isometric force of each muscle.
    If we label normalized muscle activation ``a``, and its derivative ``da``, then the penalty would evaluate at:

    .. code-block:: python

        loss = np.reduce_mean(a ** 2) + deriv_weight * np.reduce_mean(da ** 2)

    Args:
        max_iso_force: `Float` or `list`, the maximum isometric force of each muscle in the order they are declared in
            the :class:`motornet.plants.plants.Plant` object class or subclass.
        deriv_weight: `Float`, the weight of the derivative's penalty compared to the value itself.
        dt: `Float`, the size of a single timestep. This is used to calculate derivatives using Euler's method.
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensoflow` documentation for more details.
    """

    def __init__(self, max_iso_force: float, deriv_weight: float, dt: float, name: str = 'l2_xdx_activation',
                 reduction=auto_reduction):
        super().__init__(_l2_xdx_activation_loss, name=name, reduction=reduction, max_iso_force=max_iso_force,
                         deriv_weight=deriv_weight, dt=dt)


class ClippedPositionLoss(LossFunctionWrapper):
    """Applies a L1 penalty to positional error between the model's output positional state ``x`` and a user-fed
        label position ``y``:

        .. code-block:: python

            xp, _ = np.split(x, 2, axis=-1)  # remove velocity from the positional state
            yp, _ = np.split(y, 2, axis=-1)
            loss = np.reduce_mean(np.abs(xp - yp))

        If the radial distance to the desired position is less than a user-defined radius (target size), the loss is
        clipped to be 0.

        .. note::
            The positional error does not include velocity, hence the use of ``np.split`` to extract position from the
            state array.

        Args:
            target_size: `Float`, the radius around the desired position within which the position loss is clipped to 0.
            name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and
                save losses during training.
            reduction: The reduction method used. The default value is
               ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
               See the `Tensoflow` documentation for more details.
        """

    def __init__(self, target_size: float, name: str = 'position', reduction=auto_reduction):
        super().__init__(_clipped_position_loss, name=name, reduction=reduction, target_size=target_size)


def _clipped_position_loss(y_true, y_pred, target_size):
    true_pos, _ = tf.split(y_true, 2, axis=-1)
    pred_pos, _ = tf.split(y_pred, 2, axis=-1)
    err = true_pos - pred_pos
    l1 = tf.abs(err)
    l2 = tf.reduce_sum(tf.sqrt(err ** 2), axis=-1, keepdims=True)  # radial distance to desired position
    return tf.reduce_mean(tf.where(l2 < target_size, 0., l1))


def _compounded_losses(y_true, y_pred, losses, loss_weights):
    compounded_loss = loss_weights[0] * losses[0](y_true, y_pred)
    for weight, loss in zip(loss_weights[1:], losses[1:]):
        compounded_loss += weight * loss(y_true, y_pred)
    return compounded_loss


def _position_loss(y_true, y_pred):
    true_pos, _ = tf.split(y_true, 2, axis=-1)
    pred_pos, pred_vel = tf.split(y_pred, 2, axis=-1)
    # add a fixed penalty any time the arm hits the joint limits
    joint_limit_cost = tf.where(tf.equal(pred_vel[:, 1:, :], 0.), x=0., y=0.)
    return tf.reduce_mean(tf.abs(true_pos - pred_pos)) + tf.reduce_mean(joint_limit_cost)


def _l2_activation_loss(y_true, y_pred, max_iso_force):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    activation_scaled = _scale_activation(activation, max_iso_force)
    return tf.reduce_mean(activation_scaled ** 2)


# def _l2_activation_muscle_vel_loss(y_true, y_pred, max_iso_force, muscle_loss, deriv_weight):
def _l2_activation_muscle_vel_loss(y_true, y_pred, max_iso_force, deriv_weight):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    muscle_vel = tf.slice(y_pred, [0, 0, 2, 0], [-1, -1, 1, -1])
    activation_scaled = _scale_activation(activation, max_iso_force)
    return tf.reduce_mean(activation_scaled ** 2) + deriv_weight * tf.reduce_mean(muscle_vel ** 2)


def _l2_activation_l1_muscle_vel_ind_loss(y_true, y_pred, max_iso_force, activation_weight, deriv_weight):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    muscle_vel = tf.slice(y_pred, [0, 0, 2, 0], [-1, -1, 1, -1])
    activation_scaled = _scale_activation(activation, max_iso_force)
    return activation_weight * tf.reduce_mean(tf.square(activation_scaled)) + \
           deriv_weight * tf.reduce_mean(tf.abs(muscle_vel))


def _l2_xdx_activation_loss(y_true, y_pred, max_iso_force, deriv_weight, dt):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    activation_scaled = _scale_activation(activation, max_iso_force)
    d_activation = tf.reduce_mean(
        tf.square((activation_scaled[:, :, 1:, :, :] -
                   activation_scaled[:, :, :-1, :, :]) / dt)
    )
    return tf.reduce_mean(tf.square(activation_scaled)) + deriv_weight * d_activation


def _recurrent_activity_loss(y_true, y_pred, network, recurrent_weight, activity_weight):
    w = network.layers.weights[1]
    z, r, h = tf.split(w, 3, axis=1)
    r_shaped = tf.reshape(tf.transpose(y_pred, perm=[2, 1, 0]), [y_pred.shape[2], -1])
    d_r = tf.transpose(tf.reduce_sum(tf.square(_d_tanh(r_shaped)), axis=1))
    f_z = tf.reduce_sum(tf.square(z), axis=0)
    f_r = tf.reduce_sum(tf.square(r), axis=0)
    f_h = tf.reduce_sum(tf.square(h), axis=0)
    norm_val = 1 / (r_shaped.shape[1] * h.shape[0])
    f_penalty = norm_val * (tf.tensordot(d_r, f_z, 1) + tf.tensordot(d_r, f_r, 1) + tf.tensordot(d_r, f_h, 1))
    act_penalty = tf.reduce_mean(tf.square(y_pred))
    return recurrent_weight * f_penalty + activity_weight * act_penalty


def _scale_activation(activation, max_iso_force):
    """Scale activation penalty by maximum muscle force of each muscle."""
    max_iso_force_n = max_iso_force / tf.reduce_mean(max_iso_force)
    return activation * tf.expand_dims(tf.expand_dims(max_iso_force_n, axis=0), axis=0)


def _l2_regularizer(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred))


def _l2_xdx_regularizer(y_true, y_pred, deriv_weight, dt):
    dy = (y_pred[:, 1:, :] - y_pred[:, :-1, :]) / dt
    l2x = tf.reduce_mean(tf.square(y_pred))
    l2dx = tf.reduce_mean(tf.square(dy))
    return l2x + deriv_weight * l2dx


@tf.function
def _d_tanh(r):
    return 1. - tf.square(r)
