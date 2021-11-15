import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper



class L2Regularizer(LossFunctionWrapper):
    def __init__(self, name='l2_regularizer', reduction=losses_utils.ReductionV2.AUTO):
        super().__init__(y_pred_l2, name=name, reduction=reduction)


class PositionLoss(LossFunctionWrapper):
    def __init__(self, name='position', reduction=losses_utils.ReductionV2.AUTO):
        super().__init__(position_loss, name=name, reduction=reduction)


class ActivationSquaredLoss(LossFunctionWrapper):
    def __init__(self, max_iso_force, name='activation_sq', reduction=losses_utils.ReductionV2.AUTO):
        super().__init__(activation_squared_loss, name=name, reduction=reduction, max_iso_force=max_iso_force)
        self.max_iso_force = max_iso_force


class ActivationVelocitySquaredLoss(LossFunctionWrapper):
    def __init__(self, max_iso_force, vel_weight, name='activation_vel_sq', reduction=losses_utils.ReductionV2.AUTO):
        fn = activation_velocity_squared_loss
        super().__init__(fn, name=name, reduction=reduction, max_iso_force=max_iso_force, vel_weight=vel_weight)


class ActivationDiffSquaredLoss(LossFunctionWrapper):
    def __init__(self, max_iso_force, vel_weight, dt, name='d_activation_sq', reduction=losses_utils.ReductionV2.AUTO):
        fn = activation_diff_squared_loss
        super().__init__(fn, name=name, reduction=reduction, max_iso_force=max_iso_force, vel_weight=vel_weight, dt=dt)


def position_loss(y_true, y_pred):
    true_pos, _ = tf.split(y_true, 2, axis=-1)
    pred_pos, _ = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.abs(true_pos - pred_pos))


def activation_squared_loss(y_true, y_pred, max_iso_force):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    activation_scaled = scale_activation(activation, max_iso_force)
    return tf.reduce_mean(activation_scaled ** 2)


def activation_velocity_squared_loss(y_true, y_pred, max_iso_force, vel_weight):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    muscle_vel = tf.slice(y_pred, [0, 0, 2, 0], [-1, -1, 1, -1])
    activation_scaled = scale_activation(activation, max_iso_force)
    return tf.reduce_mean(activation_scaled ** 2) + vel_weight * tf.reduce_mean(tf.abs(muscle_vel))


def activation_diff_squared_loss(y_true, y_pred, max_iso_force, vel_weight, dt):
    activation = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
    activation_scaled = scale_activation(activation, max_iso_force)
    d_activation = tf.reduce_mean(tf.math.subtract(activation_scaled[1:], activation_scaled[:-1]) ** 2) * dt
    return tf.reduce_mean(activation_scaled ** 2) + vel_weight * d_activation


def scale_activation(activation, max_iso_force):
    # scale activation penalty by maximum muscle force of each muscle
    max_iso_force_n = max_iso_force / tf.reduce_mean(max_iso_force)
    return activation * tf.expand_dims(tf.expand_dims(max_iso_force_n, axis=0), axis=0)


def y_pred_l2(y_true, y_pred):
    return tf.sqrt(y_pred ** 2)

def empty_loss():
    def loss(y_true, y_pred):
        return tf.reduce_sum(tf.zeros_like(y_true))
    return loss


def position_loss():
    def loss(y_true, y_pred):
        true_pos, _ = tf.split(y_true, 2, axis=-1)
        pred_pos, _ = tf.split(y_pred, 2, axis=-1)
        return tf.reduce_mean(tf.abs(true_pos - pred_pos))
    return loss

def position_loss_bis():
    def loss(y_true, y_pred):
        true_pos,_ = tf.split(y_true, 2, axis=-1)
        pred_pos,_ = tf.split(y_pred, 2, axis=-1)
        return tf.reduce_mean(tf.norm(true_pos - pred_pos))
    return loss

def activation_squared_loss(max_iso_force):
    def loss(y_true, y_pred):
        activations = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
        # normalize max muscle force
        max_iso_force_norm = max_iso_force / tf.reduce_mean(max_iso_force)
        # scale activation penalty by maximum muscle force of each muscle
        activations_scaled = activations * tf.tile(tf.expand_dims(max_iso_force_norm, axis=2),
                                                   (tf.shape(activations)[0], tf.shape(activations)[1], 1, 1))
        return tf.reduce_mean(activations_scaled ** 2)
    return loss


def activation_velocity_squared_loss(max_iso_force, vel_weight=0.):
    def loss(y_true, y_pred):
        activations = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
        muscle_vel = tf.slice(y_pred, [0, 0, 2, 0], [-1, -1, 1, -1])
        # normalize max muscle force
        max_iso_force_norm = max_iso_force / tf.reduce_mean(max_iso_force)
        # scale activation penalty by maximum muscle force of each muscle
        activations_scaled = activations * tf.tile(tf.expand_dims(max_iso_force_norm, axis=2),
                                                   (tf.shape(activations)[0], tf.shape(activations)[1], 1, 1))
        return tf.reduce_mean(activations_scaled ** 2) + vel_weight*tf.reduce_mean(tf.abs(muscle_vel))
    return loss


def activation_diff_squared_loss(max_iso_force, vel_weight=0.):
    def loss(y_true, y_pred):
        activations = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
        # normalize max muscle force
        max_iso_force_norm = max_iso_force / tf.reduce_mean(max_iso_force)
        # scale activation penalty by maximum muscle force of each muscle
        activations_scaled = activations * tf.tile(tf.expand_dims(max_iso_force_norm, axis=2),
                                                   (tf.shape(activations)[0], tf.shape(activations)[1], 1, 1))
        activation_vel = tf.reduce_mean(tf.math.subtract(activations_scaled[1:], activations_scaled[0:-1])**2)*100
        return tf.reduce_mean(activations_scaled ** 2) + vel_weight*activation_vel
    return loss
