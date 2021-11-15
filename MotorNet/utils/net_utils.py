import tensorflow as tf


def grad(model, model_inputs, y_true, loss):
    with tf.GradientTape() as tape:
        model_pred = model(model_inputs, training=False)
        # y_pred = model_pred['cartesian position']
        loss_value = loss(y_true=y_true, y_pred=model_pred)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
