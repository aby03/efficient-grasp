from tensorflow import keras
import tensorflow as tf
# import math

# import numpy as np
def grasp_loss_bt(batch_sz = 1):
    # @tf.function
    def _grasp_loss_bt(y_true, y_pred):
        '''
        y_true: (batch, 30, 6)
        y_pred: (batch, 30, 6)
        '''
        loss = keras.backend.square(y_true - y_pred)
        loss = keras.backend.mean(loss, axis=2)
        loss = keras.backend.min(loss, axis=1)

        return loss
    return _grasp_loss_bt    

def grasp_loss(y_true, y_pred):
    theta_err_wt = 1
    y_err = y_true[0] - y_pred[0]
    x_err = y_true[1] - y_pred[1]
    
    sin_err = y_true[2] - y_pred[2]
    cos_err = y_true[3] - y_pred[3]

    h_err = y_true[4] - y_pred[4]
    w_err = y_true[5] - y_pred[5]
    # loss = x_err ** 2 + y_err ** 2 + theta_err_wt * (theta_err**2) + h_err ** 2 + w_err ** 2
    loss = (y_err ** 2 + x_err ** 2 + sin_err**2 + cos_err**2 + h_err ** 2 + w_err ** 2)/6
    return loss