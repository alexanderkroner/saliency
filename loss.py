import tensorflow as tf


def kld(y_true, y_pred, eps=1e-7):
    """This function computes the Kullback-Leibler divergence between ground
       truth saliency maps and their predictions. Values are first divided by
       their sum for each image to yield a distribution that adds to 1.

    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                       instabilities. Defaults to 1e-7.

    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """

    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keep_dims=True)
    y_true /= eps + sum_per_image

    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keep_dims=True)
    y_pred /= eps + sum_per_image

    loss = y_true * tf.log(eps + y_true / (eps + y_pred))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

    return loss
