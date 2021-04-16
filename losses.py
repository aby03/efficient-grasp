from tensorflow import keras
import tensorflow as tf
# import math

# import numpy as np
def grasp_loss_bt(batch_sz = 1):
    # @tf.function
    def _grasp_loss_bt(y_true, y_pred):
        '''
        y_true: (batch, no of labels, 6)
        y_pred: (batch, 6)
        '''
        print(tf.shape(y_true))
        loss_l = []
        for i in range(batch_sz):
            loss = float('inf')
            # for j in range(30):
            #     loss = keras.backend.minimum( keras.backend.mean( keras.backend.square(y_true[i][j] - y_pred[i]) ), loss )
            loss = keras.backend.mean(keras.backend.square(y_true[i] - y_pred[i]))
            loss_l.append(loss)
            # tf.print('DEB: ', loss_l)

        return loss_l
        # x_err = y_true[:, 0] - y_pred[:,0]
        # y_err = y_true[:, 1] - y_pred[:,1]
        # # theta_err = ((tf.math.atan(y_true[:, 2]) - tf.math.atan(y_pred[:,2])) * 180 / math.pi) % 180
        # # theta_err = tf.minimum(theta_err, 180 - theta_err)
        # sin_err = y_true[:,2] - y_pred[:,2]
        # cos_err = y_true[:,3] - y_pred[:,3]
        # h_err = y_true[:, 4] - y_pred[:,4]
        # w_err = y_true[:, 5] - y_pred[:,5]
        # loss = x_err ** 2 + y_err ** 2 + theta_err_wt * (sin_err**2 + cos_err**2) + h_err ** 2 + w_err ** 2
        # loss = x_err ** 2 + y_err ** 2 + theta_err_wt * (theta_err**2) + h_err ** 2 + w_err ** 2
        # return loss
    return _grasp_loss_bt    
