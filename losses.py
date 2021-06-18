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

# # For multi grasp model
# def grasp_loss_multi(batch_sz = 1):
#     # @tf.function
#     def _grasp_loss_multi(y_true, y_pred):
#         '''
#         y_true: (batch, 100, 7)
#         y_pred: (batch, 30, 6)
#         '''
#         yt = np.asarray(y_true)
#         loss = keras.backend.square(y_true - y_pred)
#         loss = keras.backend.mean(loss, axis=2)
#         loss = keras.backend.min(loss, axis=1)

#         return loss
#     return _grasp_loss_multi
import numpy as np
def ext_function(y_true, y_pred):
    
    # yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    # yt = np.mean(yt, axis=2)
    # yt = tf.convert_to_tensor(yt)
    yt = tf.math.reduce_mean(y_true, axis=2)
    yt = keras.backend.mean( yt, axis=1)
    print('T1: ', yt.shape)
    return yt

    # loss = keras.backend.mean(keras.backend.mean(y_pred, axis=2), axis=1)
    # return loss
    # print('T1: ', yt.shape)
    # print('P1: ', yp.shape)
    # [yp_grasps, yp_score] = np.split(yp, [6], axis=2)   
    # print('PG2: ', yp_grasps.shape)
    # print('PS2: ', yp_score.shape)
    # yp_grasps = yp_grasps[:,:,np.newaxis,:]     # (b, 100, 6) -> (b, 100, 1, 6)
    # yp_grasps = np.repeat(yp_grasps, 30, 2)     # (b, 100, 1, 6) -> (b, 100, 30, 6)

    # yt = yt[:,np.newaxis,:,:]       # (b, 30, 6) -> (b, 1, 30, 6)
    # yt = np.repeat(yt, 100, 1)      # (b, 1, 30, 6) -> (b, 100, 30, 6)

    # print('PG3: ', yp_grasps.shape)
    # print('T3: ', yt.shape)
    # grasp_loss = np.min(np.mean(np.square(yp_grasps - yt), axis=3), axis=2)
    # # print('C: ', grasp_loss.shape)
    # # print(grasp_loss[0,1])
    # SCALE=0.007
    # OFFSET=3.5
    # f_grasp_loss = SCALE*grasp_loss-OFFSET
    # print('F: ', f_grasp_loss.shape)
    # print('Fmin: ', np.min(f_grasp_loss))
    # print('Fd:', f_grasp_loss.dtype)
    # f_grasp_loss = np.exp(f_grasp_loss)
    # f_grasp_loss = 1+f_grasp_loss
    # f_grasp_loss = np.reciprocal(f_grasp_loss)
    # # f_grasp_loss = np.reciprocal(1+np.exp(SCALE*grasp_loss-OFFSET))
    # # f_grasp_loss = np.reciprocal(1+np.exp(SCALE*grasp_loss-OFFSET))
    # # print('C: ', f_grasp_loss.shape)
    # # print(f_grasp_loss[0,1])
    # yp_score = np.squeeze(yp_score, axis=2)
    # total_loss = grasp_loss + (grasp_loss - f_grasp_loss) ** 2
    # total_loss = tf.math.reduce_mean(total_loss, axis=1)
    # # print('total_loss: ', total_loss.shape)
    # # for i in range(np.shape(y_true)[1]):
    # #     print('B: ', i)
    # return total_loss

# For multi grasp model
def grasp_loss_multi(batch_sz = 1):
    # @tf.function
    def _grasp_loss_multi(y_true, y_pred):
        '''
        y_pred: (batch, 100, 7)
        y_true: (batch, 30, 6)
        '''
        SCALE=5.0
        ## CHANGE SCALE ACC TO NORMALIZATION
        # SCORE_LOSS_SCALE=1000.0
        SCORE_LOSS_SCALE=5.0
        OFFSET=3.5

        [yp_grasps, yp_score] = tf.split(y_pred, [6, 1], axis=2)

        yp_grasps = tf.expand_dims(yp_grasps, axis=2)     # (b, 100, 6) -> (b, 100, 1, 6)
        yp_grasps = tf.repeat(yp_grasps, tf.shape(y_true)[1], 2)     # (b, 100, 1, 6) -> (b, 100, 30, 6)
        # yp_grasps = tf.repeat(yp_grasps, 30, 2)     # (b, 100, 1, 6) -> (b, 100, 30, 6)

        yp_score = tf.squeeze(yp_score, axis=2)     #(b, 100, 1) -> (b,100)

        yt = tf.expand_dims(y_true, axis=1)     # (b, 30, 6) -> (b, 1, 30, 6)
        yt = tf.repeat(yt, tf.shape(y_pred)[1], 1)      # (b, 1, 30, 6) -> (b, 100, 30, 6)
        # yt = tf.repeat(yt, 100, 1)      # (b, 1, 30, 6) -> (b, 100, 30, 6)

        grasp_loss = tf.reduce_min(tf.reduce_mean(tf.square( tf.subtract( yt, yp_grasps ) ), axis=3 ), axis=2)

        f_grasp_loss = SCALE*grasp_loss-OFFSET
        f_grasp_loss = tf.clip_by_value(f_grasp_loss, -7, 7)
        f_grasp_loss = tf.exp(f_grasp_loss)
        f_grasp_loss = tf.add(1.0,f_grasp_loss)
        f_grasp_loss = tf.math.reciprocal(f_grasp_loss)

        total_loss = tf.add( SCORE_LOSS_SCALE*tf.square( tf.subtract(yp_score, f_grasp_loss) ) ,grasp_loss)
        loss = tf.math.reduce_mean(total_loss, axis=1)
        # loss = tf.math.reduce_mean(grasp_loss, axis=1)
        return loss
    return _grasp_loss_multi

# For Validation in eval callback
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