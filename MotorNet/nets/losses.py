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
