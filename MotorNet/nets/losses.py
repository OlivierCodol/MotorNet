import tensorflow as tf


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


def activation_squared_loss():
    def loss(y_true, y_pred):
        activations = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
        return tf.reduce_mean(activations ** 2)
    return loss


def activation_velocity_squared_loss():
    def loss(y_true, y_pred):
        activations = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, 1, -1])
        muscle_vel = tf.slice(y_pred, [0, 0, 2, 0], [-1, -1, 1, -1])
        return tf.reduce_mean(activations ** 2) + 0.0*tf.reduce_mean(tf.abs(muscle_vel))  # 0.05 for vel best
    return loss
