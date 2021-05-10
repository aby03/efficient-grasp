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
