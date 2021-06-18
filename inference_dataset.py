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

import json
import numpy as np
from tensorflow import keras
from dataset_processing.grasp import Grasp
from dataset_processing.cornell_generator import CornellDataset
from dataset_processing.amazon_generator import AmazonDataset

import matplotlib
import matplotlib.pyplot as plt


#######################################################
# Build Model
model, prediction_model, all_layers = build_EfficientGrasp_multi(0,
                                                        print_architecture=False)

# load pretrained weights
# model.load_weights('checkpoints/2021_05_28_03_40_07/cornell_best_grasp_accuracy.h5', by_name=True) ##101 (Final weights are not fully stored)
# model.load_weights('checkpoints/2021_05_28_22_24_04/cornell_finish.h5', by_name=True) ##102
# model.load_weights('checkpoints/2021_06_05_23_27_08/amazon_best_grasp_accuracy.h5', by_name=True) ##201(DONT USE)
# model.load_weights('checkpoints/2021_06_05_23_27_08/amazon_finish.h5', by_name=True) ##201
# model.load_weights('checkpoints/2021_06_07_03_48_48/amazon_finish.h5', by_name=True) ##202
# model.load_weights('checkpoints/2021_06_11_06_48_40/amazon_finish.h5', by_name=True) ##203 (no useful results)
# model.load_weights('checkpoints/2021_06_10_03_38_02/cornell_finish.h5', by_name=True) ##301 
model.load_weights('checkpoints/2021_06_12_16_37_53/cornell_finish.h5', by_name=True) ##TEST
print("Weights loaded!")

dataset_name = "cornell"
run_dataset = True
SAVE_FIGURE = True

if run_dataset:
    if dataset_name == "cornell":
        # Load list of images
        dataset = '/home/aby/Workspace/Cornell/archive'
        # with open(dataset+'/train_overfit_2.txt', 'r') as filehandle:
        with open(dataset+'/valid_1.txt', 'r') as filehandle:
        # with open(dataset+'/amazon_test.txt', 'r') as filehandle: ## To check Cornell trained model on amazon images
            train_data = json.load(filehandle)
        
        train_generator = CornellDataset(
            dataset,
            train_data,
            train=False,
            shuffle=False,
            batch_size=1
        )
    else:
        dataset = '/home/aby/Workspace/parallel-jaw-grasping-dataset/data'
        with open(dataset+'/test-split.txt', 'r') as filehandle:
            lines = filehandle.readlines()
            train_data = []
            for line in lines:
                train_data.append(line.strip())
        
        train_generator = AmazonDataset(
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
        if dataset_name == "cornell":
            rgd_img = image.Image.from_file(dataset+train_data[i])
        else:
            rgd_img = image.Image.from_file(dataset+'/heightmap-color/'+train_data[i]+'.png')
        init_shape = rgd_img.img.shape
        ## Perform crop to min(height,width)
        left = max(0, init_shape[1] - init_shape[0])//2
        top = max(0, init_shape[0] - init_shape[1])//2
        top_left = (top, left)
        bottom_right = (init_shape[0]-top, init_shape[1]-left)
        rgd_img.crop(top_left, bottom_right)
        ## Perform central zoom
        if dataset_name == "cornell":
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
        
        DISPLAY_PRED = 100
        # Sort y_out based on score
        sort_index = (-all_score).argsort()
        all_score = all_score[sort_index]
        all_predictions = all_predictions[sort_index,:]
        print(all_score)
        # print(all_score[:DISPLAY_PRED])

        ### Plot all grasps
        fig, ax = plt.subplots(2,2)
        fig.set_size_inches(10, 9)
        ## Show Original Image
        ax[0][0].imshow(disp_img)
        ax[0][0].set_title('Original RGD Image')
        ## Show RGD image and set axis title
        ax[0][1].imshow(X[0,:,:,:])
        ax[0][1].set_title('Labelled Grasps on RGD Image')
        ax[1][0].imshow(disp_img)
        ax[1][0].set_title('Predicted Grasps with score')
        ax[1][1].imshow(disp_img)
        ax[1][1].set_title('Predicted and Labelled grasps')
        ## Show labelled grasps on orig image
        lab_grasp = Y_true[0]
        for j in range(len(lab_grasp)):
            if not lab_grasp[j][0] == 1e8:
                plot_grasp = Grasp(lab_grasp[j][0:2], *lab_grasp[j][2:], unnorm=True)
                plot_grasp.plot(ax[0][1], 'green')
                plot_grasp.plot(ax[1][1], 'green')
        ## Show Predicted grasps on orig image
        colormap=plt.get_cmap('cool')
        display_score = all_score[:DISPLAY_PRED]
        center_y = []
        center_x = []
        col_norm = matplotlib.colors.Normalize(vmin=display_score[-1], vmax=display_score[0], clip=False)
        for j in range(min(all_predictions.shape[0], DISPLAY_PRED)):
            plot_grasp = Grasp(all_predictions[j][0:2], *all_predictions[j][2:], quality=display_score[j], unnorm=True)
            plot_points = plot_grasp.as_gr.points
            points = np.vstack((plot_points, plot_points[0]))
            ax[1][0].plot(points[:, 1], points[:, 0], color=colormap(col_norm(display_score[j])))
            ax[1][1].plot(points[:, 1], points[:, 0], color=colormap(col_norm(display_score[j])))
            center_y.append(plot_grasp.center[0])
            center_x.append(plot_grasp.center[1])
        sc=ax[1][0].scatter(center_x, center_y, c=display_score, cmap=colormap)
        ax[1][1].scatter(center_x, center_y, c=display_score, cmap=colormap)
        plt.colorbar(sc)
        if SAVE_FIGURE:
            fname = '/home/aby/Pictures/'+str(i)
            plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                frameon=None, metadata=None)
        plt.show()      
else:
    ## TEST ON NON LABELLED IMAGES
    # Load list of images
    dataset = '/home/aby/Workspace/Cornell/archive'
    with open(dataset+'/amazon_test.txt', 'r') as filehandle: ## To check trained model on unlabelled unprocessed amazon images
        train_data = json.load(filehandle)
    # Load Image
    for i in range(len(train_data)):
        X = CornellDataset.load_custom_image(dataset+train_data[i])
        disp_img = CornellDataset.load_custom_image(dataset+train_data[i], normalise=False)
        # Expand dim for batch
        X = X[np.newaxis, ...]
        Y_pred = model.predict(X)

        # Remove batch dim
        Y_out = Y_pred[0,:,:]

        all_predictions = Y_out[:,0:6]
        all_score = Y_out[:,6]
        
        DISPLAY_PRED = 10
        # Sort y_out based on score
        sort_index = (-all_score).argsort()
        all_score = all_score[sort_index]
        all_predictions = all_predictions[sort_index,:]
        print(all_score[:DISPLAY_PRED])

        ### Plot all grasps
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(10, 5)
        ## Show Original Image
        ax[0].imshow(disp_img)
        ax[0].set_title('Original Image')
        ## Show RGD image and set axis title
        ax[1].imshow(disp_img)
        ax[1].set_title('Predicted Grasps')
        ## Show Predicted grasps on orig image
        colormap=plt.get_cmap('cool')
        display_score = all_score[:DISPLAY_PRED]
        center_y = []
        center_x = []
        col_norm = matplotlib.colors.Normalize(vmin=display_score[-1], vmax=display_score[0], clip=False)
        for j in range(min(all_predictions.shape[0], DISPLAY_PRED)):
            plot_grasp = Grasp(all_predictions[j][0:2], *all_predictions[j][2:], quality=display_score[j], unnorm=True)
            plot_points = plot_grasp.as_gr.points
            points = np.vstack((plot_points, plot_points[0]))
            ax[1].plot(points[:, 1], points[:, 0], color=colormap(col_norm(display_score[j])))
            center_y.append(plot_grasp.center[0])
            center_x.append(plot_grasp.center[1])
        sc=ax[1].scatter(center_x, center_y, c=display_score, cmap=colormap)
        # ax[1][1].scatter(center_x, center_y, c=display_score, cmap=colormap)
        plt.colorbar(sc)
        if SAVE_FIGURE:
            fname = '/home/aby/Pictures/'+str(i)
            plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                frameon=None, metadata=None)
        plt.show()

        # # Plot predicted grasp
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        # ax.imshow(disp_img)
        # pred_grasp.plot(ax, 'red')
        # plt.show()
