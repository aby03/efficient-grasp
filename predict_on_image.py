import tensorflow as tf
from dataset_processing import grasp

def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true
    """
    # Eager execution
    # tf.compat.v1.enable_eager_execution()
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # _ = tf.Session(config = config)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Set before importing and running
allow_gpu_growth_memory()

# Import rest
from model_multi import build_EfficientGrasp_multi
# from model_split import build_EfficientGrasp
import json
import numpy as np
from tensorflow import keras
from dataset_processing.grasp import Grasp
from dataset_processing.cornell_generator import CornellDataset

import matplotlib.pyplot as plt

# Build Model
model, prediction_model, all_layers = build_EfficientGrasp_multi(0,
                                                        print_architecture=False)

# load pretrained weights
model.load_weights('checkpoints/2021_05_21_01_20_20/cornell_best_grasp_accuracy.h5', by_name=True)
print("Weights loaded!")


run_multiple = True
if run_multiple:
    # Load list of images
    dataset = '/home/aby/Workspace/MTP/Datasets/Cornell/archive'
    with open(dataset+'/test.txt', 'r') as filehandle:
        train_data = json.load(filehandle)

    # Visualization on Custom Images
    for i, filename in enumerate(train_data):
        # Load Images
        X = CornellDataset.load_custom_image(dataset+filename)
        disp_img = CornellDataset.load_custom_image(dataset+filename, normalise=False)
        # Expand dim for batch
        X = X[np.newaxis, ...]
        Y_pred = model.predict(X)       # Pred -> (b, 100, 7) where b=1

        # Remove batch dim
        Y_out = Y_pred[0,:,:]
        all_predictions = Y_out[:,0:6]
        all_score = Y_pred[:,6]
        
        print(Y_out)
        # Plot all grasps
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.imshow(disp_img)
        for i in range(all_predictions.shape[0]):
            plot_grasp = Grasp(all_predictions[i][0:2], *all_predictions[i][2:], unnorm=True)
            plot_grasp.plot(ax, 'red')
        plt.show()


        # test_out = Y_pred[0][0]
        # print(test_out)
        # pred_grasp = Grasp(test_out[0:2], *test_out[2:], unnorm=True)

        # # Plot predicted grasp
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        # ax.imshow(disp_img)
        # pred_grasp.plot(ax, 'red')
        # plt.show()
else:
    ## TEST ON SINGLE IMAGE
    filename = '/home/aby/Workspace/MTP/Datasets/Cornell/archive/06/pcd0600r.png'
    # Load Image
    X = CornellDataset.load_custom_image(filename)
    disp_img = CornellDataset.load_custom_image(filename, normalise=False)
    # Expand dim for batch
    X = X[np.newaxis, ...]
    Y_pred = model.predict(X)
    # Remove batch dim
    test_out = Y_pred[0][0]
    pred_grasp = Grasp(test_out[0:2], *test_out[2:], unnorm=True)

    # Plot predicted grasp
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(disp_img)
    pred_grasp.plot(ax, 'red')
    plt.show()
