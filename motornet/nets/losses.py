import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper

"""
Regularizer indicates the penalization is applied to the *value* of the passed input, not the error.
xDx indicates the value and the derivative of the passed input is penalized, with the derivative loss scaled by 
    a used-defined "deriv_weight" scalar compared to the value.  
"""

auto_reduction = losses_utils.ReductionV2.AUTO


class L2xDxRegularizer(LossFunctionWrapper):
    def __init__(self, deriv_weight, dt, name='gru_regularizer', reduction=auto_reduction):
        super().__init__(l2_xdx_regularizer, name=name, reduction=reduction, deriv_weight=deriv_weight, dt=dt)


class L2Regularizer(LossFunctionWrapper):
    def __init__(self, name='l2_regularizer', reduction=auto_reduction):
        super().__init__(l2_regularizer, name=name, reduction=reduction)


class RecurrentActivityRegularizer(LossFunctionWrapper):
    def __init__(self, network, recurrent_weight, activity_weight, name='recurrent_activity',
                 reduction=auto_reduction):
        super().__init__(recurrent_activity_loss, network=network, recurrent_weight=recurrent_weight,
                         activity_weight=activity_weight, name=name, reduction=reduction)


class PositionLoss(LossFunctionWrapper):
    def __init__(self, name='position', reduction=auto_reduction):
        super().__init__(position_loss, name=name, reduction=reduction)


class L2ActivationLoss(LossFunctionWrapper):
    def __init__(self, max_iso_force, name='l2_activation', reduction=auto_reduction):
        super().__init__(l2_activation_loss, name=name, reduction=reduction, max_iso_force=max_iso_force)
        self.max_iso_force = max_iso_force


class L2ActivationMuscleVelLoss(LossFunctionWrapper):
    def __init__(self, max_iso_force, deriv_weight, name='l2_activation_muscle_vel', reduction=auto_reduction):
        super().__init__(l2_activation_muscle_vel_loss, name=name, reduction=reduction, max_iso_force=max_iso_force,
                         deriv_weight=deriv_weight)


class L2xDxActivationLoss(LossFunctionWrapper):
    def __init__(self, max_iso_force, deriv_weight, dt, name='l2_xdx_activation', reduction=auto_reduction):
        super().__init__(l2_xdx_activation_loss, name=name, reduction=reduction, max_iso_force=max_iso_force,
                         deriv_weight=deriv_weight, dt=dt)


def position_loss(y_true, y_pred):
    true_pos, _ = tf.split(y_true, 2, axis=-1)
    pred_pos, pred_vel = tf.split(y_pred, 2, axis=-1)
    # add a fixed penalty any time the arm hits the joint limits
    joint_limit_cost = tf.where(tf.equal(pred_vel[:, 1:, :], 0.), x=0., y=0.)
    return tf.reduce_mean(tf.abs(true_pos - pred_pos)) + tf.reduce_mean(joint_limit_cost)


def l2_activation_loss(y_true, y_pred, max_iso_force):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    activation_scaled = scale_activation(activation, max_iso_force)
    return tf.reduce_mean(activation_scaled ** 2)


def l2_activation_muscle_vel_loss(y_true, y_pred, max_iso_force, muscle_loss, deriv_weight):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    muscle_vel = tf.slice(y_pred, [0, 0, 2, 0], [-1, -1, 1, -1])
    activation_scaled = scale_activation(activation, max_iso_force)
    return muscle_loss * tf.reduce_mean(activation_scaled ** 2) + deriv_weight * tf.reduce_mean(tf.abs(muscle_vel))


def l2_xdx_activation_loss(y_true, y_pred, max_iso_force, deriv_weight, dt):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    activation_scaled = scale_activation(activation, max_iso_force)
    d_activation = tf.reduce_mean(tf.abs((activation_scaled[:, :, 1:, :, :] -
                                             activation_scaled[:, :, :-1, :, :]) / dt))
    return tf.reduce_mean(tf.square(activation_scaled)) + deriv_weight * d_activation


def recurrent_activity_loss(y_true, y_pred, network, recurrent_weight, activity_weight):
    w = network.layers.weights[1]
    z, r, h = tf.split(w, 3, axis=1)
    r_shaped = tf.reshape(tf.transpose(y_pred, perm=[2, 1, 0]), [y_pred.shape[2], -1])
    d_r = tf.transpose(tf.reduce_sum(tf.square(d_tanh(r_shaped)), axis=1))
    f_z = tf.reduce_sum(tf.square(z), axis=0)
    f_r = tf.reduce_sum(tf.square(r), axis=0)
    f_h = tf.reduce_sum(tf.square(h), axis=0)
    norm_val = 1 / (r_shaped.shape[1] * h.shape[0])
    f_penalty = norm_val * (tf.tensordot(d_r, f_z, 1) + tf.tensordot(d_r, f_r, 1) + tf.tensordot(d_r, f_h, 1))
    act_penalty = tf.reduce_mean(tf.square(y_pred))
    return recurrent_weight * f_penalty + activity_weight * act_penalty


def scale_activation(activation, max_iso_force):
    """scale activation penalty by maximum muscle force of each muscle"""
    max_iso_force_n = max_iso_force / tf.reduce_mean(max_iso_force)
    return activation * tf.expand_dims(tf.expand_dims(max_iso_force_n, axis=0), axis=0)


def l2_regularizer(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred))


def l2_xdx_regularizer(y_true, y_pred, deriv_weight, dt):
    dy = (y_pred[:, 1:, :] - y_pred[:, :-1, :]) / dt
    l2x = tf.reduce_mean(tf.square(y_pred))
    l2dx = tf.reduce_mean(tf.square(dy))
    return l2x + deriv_weight * l2dx


@tf.function
def d_tanh(r):
    return 1. - tf.square(r)
