import tensorflow as tf  #(version 2.30)
import tensorflow.keras.backend as K


def cox_ph_loss(y_true, y_pred):
    """
    :param y_true: n x 2 tensor: n, (time, event)
    :param y_pred: n x 1 tensor: n, beta*X 
    :return: survival loss: -sum_{i:U=1}{beta*X_i - log{sum_{j in Omega_i}_{exp(beta*X_j)}}
    """
    sum_loss = 0.0
    for i in tf.range( len( y_pred[:,0]) ):
        omega_i = tf.math.greater_equal(y_true[:, 0], y_true[i, 0])
        sum_loss += y_true[i,1] * (y_pred[i,0] - tf.math.log(1e-20 + tf.math.reduce_sum(tf.boolean_mask(K.exp(y_pred[:,0]), omega_i))))
    norm_factor = tf.math.reduce_sum(tf.cast(tf.math.greater(y_true[:, 1], 0), K.floatx()))
    
    return sum_loss*(-1)/norm_factor  #layer norm can be adjusted. 