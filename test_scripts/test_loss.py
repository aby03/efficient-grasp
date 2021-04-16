import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
# tf.enable_eager_execution()
# from losses import grasp_loss_bt
from tensorflow import keras
def grasp_loss_bt(batch_sz = 1):
    # @tf.function
    def _grasp_loss_bt(y_true, y_pred):
        '''
        y_true: (batch, no of labels, 6)
        y_pred: (batch, 6)
        '''
        # print(tf.shape(y_true))
        # y_true_unpacked = tf.unstack(y_true)
        # y_true.set_shape([batch_sz, 30, 6])
        loss_l = []
        for i in range(batch_sz):
            # y_true_g_unpacked = tf.unstack(y_true_unpacked[i])
            loss = float('inf')
            for j in range(2):
                # if tf.equal(y_true[i][j][0],  tf.constant(0)):
                #     break
                # tf.print(y_pred)
                # tf.print('*******')
                # tf.print(y_true[i,j])
                loss = keras.backend.minimum( keras.backend.mean( keras.backend.square(y_true[i][j] - y_pred[i]) ), loss )
                # tf.print('******')
                # tf.print(loss)
                # tf.print('=============')
                # loss = keras.backend.minimum( keras.backend.sum((y_true[i][j] - y_pred[i])**2), loss )
                # loss = min( np.sum( (y_true[i,j,:] - y_pred[i,:])**2 ), loss)
            # tf.print('##########')
            loss_l.append(loss)
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

test_func = grasp_loss_bt(1)
# 2, 2, 6
x = [   #I1
        [   [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0]  ],
        #I2
        [   [1, 2, 3, 4, 0, 0], [1, 2, 3, 0, 0, 0]  ]
    ] 
y = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]  
F = test_func(K.variable(x), K.variable(y))
p = K.eval(F)
print(len(p))
print(p)
tf.print(p)