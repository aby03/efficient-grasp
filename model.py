# from functools import reduce

# import tensorflow as tf
from tensorflow.keras import layers
# from tensorflow.keras import initializers
from tensorflow.keras import models
# from tensorflow.keras import backend
# from tensorflow.keras import regularizers
# from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

# from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization, RegressTranslation, CalculateTxTy, GroupNormalization
# from initializers import PriorProbability
# # from utils.anchors import anchors_for_shape
# import numpy as np


# MOMENTUM = 0.997
# EPSILON = 1e-4


def build_EfficientGrasp(phi,
                        num_anchors = 1,
                        freeze_bn = False,
                        anchor_parameters = None,
                        print_architecture = False):
    # Parameters for model
    input_shape = (512, 512, 3)
    output_dim = 6
    # Input layer
    image_input = layers.Input(input_shape)
    
    # Debug Architecture
    grasp_regression = layers.MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='valid')(image_input)
    grasp_regression = layers.Flatten()(grasp_regression)

    # Final Layer into 6 dim vector
    grasp_regression = layers.Dense(output_dim, name='regression')(grasp_regression)

    efficientgrasp_train = models.Model(inputs = [image_input], outputs = grasp_regression, name = 'efficientgrasp')
    
    if print_architecture:
        efficientgrasp_train.summary()
    # Return Model
    return efficientgrasp_train

# build_EfficientGrasp(0, print_architecture=True)