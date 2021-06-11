import tensorflow as tf
from dataset_processing import grasp, image

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

import matplotlib
import matplotlib.pyplot as plt


#######################################################
# Build Model
model, prediction_model, all_layers = build_EfficientGrasp_multi(0,
                                                        print_architecture=False)

# load pretrained weights
# model.load_weights('checkpoints/2021_05_28_03_40_07/cornell_best_grasp_accuracy.h5', by_name=True)
model.load_weights('checkpoints/2021_05_28_03_40_07/cornell_finish.h5', by_name=True)
print("Weights loaded!")


run_multiple = True
if run_multiple:
    # Load list of images
    dataset = '/home/aby/Workspace/Cornell/archive'
    # with open(dataset+'/test.txt', 'r') as filehandle:
    with open(dataset+'/valid_1.txt', 'r') as filehandle:
    # with open(dataset+'/amazon_test.txt', 'r') as filehandle:
        train_data = json.load(filehandle)

    train_generator = CornellDataset(
        dataset,
        train_data,
        train=False,
        shuffle=False,
        batch_size=1
    )

    # Visualization on Custom Images
    for i in range(len(train_generator)):
        # Load Images
        X, Y_true = train_generator[i]
        # disp_img = X[0,:,:,:]

        rgd_img = image.Image.from_file(dataset+train_data[i])
        init_shape = rgd_img.img.shape
        ## Perform crop to min(height,width)
        left = max(0, init_shape[1] - init_shape[0])//2
        top = max(0, init_shape[0] - init_shape[1])//2
        top_left = (top, left)
        bottom_right = (init_shape[0]-top, init_shape[1]-left)
        rgd_img.crop(top_left, bottom_right)
        ## Perform central zoom
        rgd_img.zoom(0.875)
        ## Resizing (Side scaling)
        output_size = 512
        rgd_img.resize((output_size, output_size))
        disp_img = rgd_img.img

        ## Predict on image
        Y_pred = model.predict(X)       # Pred -> (b, 100, 7) where b=1

        # Remove batch dim
        Y_out = Y_pred[0,:,:]

        all_predictions = Y_out[:,0:6]
        all_score = Y_out[:,6]
        
        DISPLAY_PRED = 10
        # Sort y_out based on score
        sort_index = (-all_score).argsort()
        print(all_score.shape)
        print(all_predictions.shape)
        all_score = all_score[sort_index]
        all_predictions = all_predictions[sort_index,:]

        print(all_score[:DISPLAY_PRED])
        ### Plot all grasps
        fig, ax = plt.subplots(2,2)
        ## Show Original Image
        ax[0][0].imshow(disp_img)
        ax[0][0].set_title('Original Image')
        ## Show RGD image and set axis title
        ax[0][1].imshow(X[0,:,:,:])
        ax[0][1].set_title('Labelled Grasps on RGD Image')
        ax[1][0].imshow(X[0,:,:,:])
        ax[1][0].set_title('Predicted Grasps with score')
        ax[1][1].imshow(X[0,:,:,:])
        ax[1][1].set_title('Predicted and Labelled grasps')
        ## Show labelled grasps on orig image
        lab_grasp = Y_true[0]
        for i in range(len(lab_grasp)):
            if not lab_grasp[i][0] == 1e8:
                plot_grasp = Grasp(lab_grasp[i][0:2], *lab_grasp[i][2:], unnorm=True)
                plot_grasp.plot(ax[0][1], 'green')
                plot_grasp.plot(ax[1][1], 'green')
        ## Show Predicted grasps on orig image
        colormap=plt.get_cmap('plasma')
        display_score = all_score[:DISPLAY_PRED]
        center_y = []
        center_x = []
        col_norm = matplotlib.colors.Normalize(vmin=display_score[-1], vmax=display_score[0], clip=False)
        for i in range(min(all_predictions.shape[0], DISPLAY_PRED)):
            plot_grasp = Grasp(all_predictions[i][0:2], *all_predictions[i][2:], quality=display_score[i], unnorm=True)
            plot_points = plot_grasp.as_gr.points
            points = np.vstack((plot_points, plot_points[0]))
            ax[1][0].plot(points[:, 1], points[:, 0], color=colormap(col_norm(display_score[i])))
            ax[1][1].plot(points[:, 1], points[:, 0], color=colormap(col_norm(display_score[i])))
            center_y.append(plot_grasp.center[0])
            center_x.append(plot_grasp.center[1])
        sc=ax[1][0].scatter(center_x, center_y, c=display_score, cmap=colormap)
        ax[1][1].scatter(center_x, center_y, c=display_score, cmap=colormap)
        plt.colorbar(sc)
        plt.show()      
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
