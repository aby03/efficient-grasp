import os

from tensorflow.python.util.nest import yield_flat_paths
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
# tf.enable_eager_execution()
# from losses import grasp_loss_bt
from tensorflow import keras

def ext_function(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    [yt_grasps, yt_score] = np.split(yt, [6], axis=2)   

    yt_grasps = yt_grasps[:,:,np.newaxis,:]     # (b, 100, 6) -> (b, 100, 1, 6)
    yt_grasps = np.repeat(yt_grasps, 30, 2)     # (b, 100, 1, 6) -> (b, 100, 30, 6)

    yp = yp[:,np.newaxis,:,:]       # (b, 30, 6) -> (b, 1, 30, 6)
    yp = np.repeat(yp, 100, 1)      # (b, 1, 30, 6) -> (b, 100, 30, 6)

    # print('A: ', yt_grasps.shape)
    # print('B: ', yp.shape)
    grasp_loss = np.min(np.mean(np.square(yt_grasps - yp), axis=3), axis=2)
    # print('C: ', grasp_loss.shape)
    # print(grasp_loss[0,1])
    SCALE=0.007
    OFFSET=3.5
    f_grasp_loss = np.reciprocal(1+np.exp(SCALE*grasp_loss-OFFSET))
    # print('C: ', f_grasp_loss.shape)
    # print(f_grasp_loss[0,1])
    yt_score = np.squeeze(yt_score, axis=2)
    total_loss = grasp_loss + (grasp_loss - f_grasp_loss) ** 2
    total_loss = np.mean(total_loss, axis=1)
    # print('total_loss: ', total_loss.shape)
    # for i in range(np.shape(y_true)[1]):
    #     print('B: ', i)
    return total_loss

# For multi grasp model
def grasp_loss_multi(batch_sz = 1):
    # @tf.function
    def _grasp_loss_multi(y_true, y_pred):
        '''
        y_true: (batch, 100, 7)
        y_pred: (batch, 30, 6)
        '''

        [yp_grasps, yp_score] = tf.split(y_pred, [6, 1], axis=2)

        yp_grasps = tf.expand_dims(yp_grasps, axis=2)     # (b, 100, 6) -> (b, 100, 1, 6)
        yp_grasps = tf.repeat(yp_grasps, 30, 2)     # (b, 100, 1, 6) -> (b, 100, 30, 6)

        yp_score = tf.squeeze(yp_score, axis=2)     #(b, 100, 1) -> (b,100)

        yt = tf.expand_dims(y_true, axis=1)     # (b, 30, 6) -> (b, 1, 30, 6)
        yt = tf.repeat(yt, 100, 1)      # (b, 1, 30, 6) -> (b, 100, 30, 6)

        grasp_loss = tf.reduce_min(tf.reduce_mean(tf.square( tf.subtract( yt, yp_grasps ) ), axis=3 ), axis=2)

        SCALE=0.007
        OFFSET=3.5
        f_grasp_loss = SCALE*grasp_loss-OFFSET
        f_grasp_loss = tf.clip_by_value(f_grasp_loss, 0, 5)
        f_grasp_loss = tf.exp(f_grasp_loss)
        print('f_l_val: ', f_grasp_loss)
        tf.print(f_grasp_loss)
        f_grasp_loss = tf.add(1,f_grasp_loss)
        f_grasp_loss = tf.math.reciprocal(f_grasp_loss)

        total_loss = tf.add( tf.square( tf.subtract(yp_score, f_grasp_loss) ) ,grasp_loss)
        loss = tf.math.reduce_mean(total_loss, axis=1)

        return loss
    return _grasp_loss_multi


def grasp_loss_bt(batch_sz = 1):
    @tf.function
    def _grasp_loss_bt(y_true, y_pred):
        '''
        y_true: (batch, no of labels, 6)
        y_pred: (batch, 6)
        '''
        print('True: ', tf.shape(y_true))
        print('Pred: ', tf.shape(y_pred))

        loss = keras.backend.square(y_true - y_pred)
        loss = keras.backend.mean(loss, axis=2)
        loss = keras.backend.min(loss, axis=1)
        # loss = keras.backend.mean(keras.backend.square(y_true - y_pred))
        # loss = keras.backend.mean(keras.backend.square(y_true - y_pred))

        # y_true_unpacked = tf.unstack(y_true)
        # y_true.set_shape([batch_sz, 30, 6])
        # loss_l = []
        # for i in range(batch_sz):
        #     # y_true_g_unpacked = tf.unstack(y_true_unpacked[i])
        #     loss = float('inf')

        #     loss = keras.backend.mean(keras.backend.square(y_true[i] - y_pred[i])) 

        #     # for j in range(2):
        #         # if tf.equal(y_true[i][j][0],  tf.constant(0)):
        #         #     break
        #         # tf.print(y_pred)
        #         # tf.print('*******')
        #         # tf.print(y_true[i,j])
        #         # loss = keras.backend.minimum( keras.backend.mean( keras.backend.square(y_true[i][j] - y_pred[i]) ), loss )
        #         # tf.print('******')
        #         # tf.print(loss)
        #         # tf.print('=============')
        #         # loss = keras.backend.minimum( keras.backend.sum((y_true[i][j] - y_pred[i])**2), loss )
        #         # loss = min( np.sum( (y_true[i,j,:] - y_pred[i,:])**2 ), loss)
        #     # tf.print('##########')
        #     loss_l.append(loss)
        return loss
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

test_func = grasp_loss_multi(1)
# 2, 2, 6
# x = [   #I1
#         [   [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0]  ],
#         #I2
#         [   [1, 2, 3, 4, 0, 0], [1, 2, 3, 0, 0, 0]  ]
#     ] 
# y = [
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0]
#     ] 
# print('X: ', x.shape)
x = np.zeros([2,30,6])
y = np.ones([2,100,7])
xx = K.variable(x)
yy = K.variable(y)
# print('XX: ', tf.shape(xx[0]))
F = test_func(K.variable(x), K.variable(y))
p = K.eval(F)
# print(len(p))
print(p)
# print(p[0])
print('Ans: ')
# tf.print_variable(p)
import sys
tf.print("tensors:", p, output_stream=sys.stdout)
