import tensorflow as tf
from dataset_processing import grasp, image
from shapely.geometry import Polygon # For IoU

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
from dataset_processing.vmrd_generator import VMRDDataset
from dataset_processing.grasp import get_grasp_from_pred
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

#######################################################
iou_threshold = 0.25
#######################################################
# Build Model
model, prediction_model, all_layers = build_EfficientGrasp_multi(0,
                                                        score_threshold = 0.3,
                                                        print_architecture=False)

# load pretrained weights
# model.load_weights('checkpoints/2021_05_28_03_40_07/cornell_best_grasp_accuracy.h5', by_name=True) ##101 (Final weights are not fully stored)
# model.load_weights('checkpoints/2021_05_28_22_24_04/cornell_finish.h5', by_name=True) ##102
# model.load_weights('checkpoints/2021_06_05_23_27_08/amazon_best_grasp_accuracy.h5', by_name=True) ##201(DONT USE)
# model.load_weights('checkpoints/2021_06_05_23_27_08/amazon_finish.h5', by_name=True) ##201
# model.load_weights('checkpoints/2021_06_07_03_48_48/amazon_finish.h5', by_name=True) ##202
# model.load_weights('checkpoints/2021_06_11_06_48_40/amazon_finish.h5', by_name=True) ##203 (no useful results)
# model.load_weights('checkpoints/2021_06_10_03_38_02/cornell_finish.h5', by_name=True) ##301 
# model.load_weights('checkpoints/2021_06_12_16_37_53/cornell_finish.h5', by_name=True) ##TEST

prediction_model.load_weights('checkpoints/2021_06_24_02_57_59/vmrd_best_val_loss.h5', by_name=True) ##VMRD
# prediction_model.load_weights('checkpoints/2021_06_23_20_33_49/cornell_finish.h5', by_name=True) ##CORNELL
print("Weights loaded!")

dataset_name = "vmrd"
run_dataset = True
SAVE_FIGURE = False
### to show all keep Show_plots true and show_selective_plots false
### to show select
SHOW_PLOTS = False
SHOW_SELECTIVE_PLOTS = False
selective_plots = [70, 80, 92, 97, 113, 118, 132]

if run_dataset:
    if dataset_name == "cornell":
        # Load list of images
        dataset = '/home/aby/Workspace/Cornell/archive'
        # with open(dataset+'/train_overfit_2.txt', 'r') as filehandle:
        with open(dataset+'/valid_1.txt', 'r') as filehandle:
        # with open(dataset+'/amazon_test.txt', 'r') as filehandle: ## To check Cornell trained model on amazon images
            train_data = json.load(filehandle)
        
        val_generator = CornellDataset(
            dataset,
            train_data,
            train=False,
            shuffle=False,
            batch_size=1
        )
    elif dataset_name == "vmrd":
        dataset = '/home/aby/Workspace/vmrd-v2'
        with open(dataset+'/ImageSets/Main/trainval.txt', 'r') as filehandle:
            lines = filehandle.readlines()
            data = []
            for line in lines:
                data.append(line.strip())
        train_data = []
        valid_data = []
        for i in range(len(data)):
            if not i%10 == 0:
                train_data.append(data[i])
            else:
                valid_data.append(data[i])
        val_generator = VMRDDataset(
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
        
        val_generator = AmazonDataset(
            dataset,
            train_data,
            train=False,
            shuffle=False,
            batch_size=1
        )
    true_label_list = []
    score_list = []
    detections_list = []
    ang_class_list = []
    # Visualization on Custom Images
    for i in tqdm(range(len(val_generator))):
        # Load Images
        X, Y_true = val_generator.get_annotation_val(i)
        # disp_img = X[0,:,:,:]
        if dataset_name == "cornell":
            rgd_img = image.Image.from_file(dataset+train_data[i])
        elif dataset_name == "vmrd":
            rgd_img = image.Image.from_file(dataset+'/JPEGImages/'+train_data[i]+'.jpg')
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
        Y_pred = prediction_model.predict(X)       # Pred -> (b, 100, 7) where b=1

        # Remove batch dim
        # Y_out = Y_pred[0]
        # print('DEBG: ', Y_out.shape)
        # print('DEBG: ', Y_pred[0].shape)
        # print('DEBG: ', Y_pred[1].shape)
        # print('DEBG: ', Y_pred[2].shape)
        all_predictions_bbox = Y_pred[0][0]
        all_score = Y_pred[1][0]
        all_angle_class = Y_pred[2][0]
        
        # Sort y_out based on score
        sort_index = (-all_score).argsort()
        all_score = all_score[sort_index]
        all_predictions_bbox = all_predictions_bbox[sort_index,:]
        all_angle_class = all_angle_class[sort_index]
        # # Remove ignore indices
        retain_indices, = np.where(all_score != -1.0)
        all_score = all_score[retain_indices]
        all_predictions_bbox = all_predictions_bbox[retain_indices,:]
        all_angle_class = all_angle_class[retain_indices]
        # Store for eval
        score_list.append(all_score)
        detections_list.append(all_predictions_bbox)
        ang_class_list.append(all_angle_class)
        true_label_list.append(Y_true)

        if SHOW_PLOTS:
            if SHOW_SELECTIVE_PLOTS and i not in selective_plots:
                continue
            DISPLAY_PRED = 100
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
            # print(lab_grasp)
            for j in range(lab_grasp['bboxes'].shape[0]):
                plot_grasp = get_grasp_from_pred(lab_grasp['bboxes'][j], lab_grasp['labels'][j])
                # plot_grasp = Grasp(lab_grasp[j][0:2], *lab_grasp[j][2:], unnorm=True)
                plot_grasp.plot(ax[0][1], 'green')
                plot_grasp.plot(ax[1][1], 'green')
            ## Show Predicted grasps on orig image
            colormap=plt.get_cmap('cool')
            display_score = all_score[:DISPLAY_PRED]
            center_y = []
            center_x = []
            col_norm = matplotlib.colors.Normalize(vmin=display_score[-1], vmax=display_score[0], clip=False)
            for j in range(min(all_predictions_bbox.shape[0], DISPLAY_PRED)):
                plot_grasp = get_grasp_from_pred(all_predictions_bbox[j], all_angle_class[j])
                # plot_grasp = Grasp(all_predictions[j][0:2], *all_predictions[j][2:], quality=display_score[j], unnorm=True)
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
    ##################################################################################################
    # Eval Code
    ##################################################################################################
    ## For metrics
    iou_list = []
    angle_diff_list = []
    pred_count = 0
    correct_pred_count = 0      # Total Predictions Count
    correct_img_pred_count = 0  # only 1 prediction counted per image
    avg_pred_grasp_score = 0
    top_score_correct = 0
    incorrect_top_score = [] # To find images with bad top score grasp
    # For each image in batch
    for i in range(len(detections_list)):
        img_correct_pred = False
        # For each pred grasp
        for j in range(detections_list[i].shape[0]):
            correct_pred = False
            # Create predicted grasp in right format
            pred_grasp_bbox = detections_list[i][j]    # xmin, ymin, xmax, ymax
            pred_grasp_score = score_list[i][j]
            pred_grasp_angle_class = ang_class_list[i][j]
            pred_grasp = get_grasp_from_pred(pred_grasp_bbox, pred_grasp_angle_class)
            # For avg pred score calc
            avg_pred_grasp_score += pred_grasp_score
            # For IoU calculation
            bbox_pred = pred_grasp.as_bbox
            # Counting total grasp predictions
            pred_count += 1
            # For each true_grasp
            for k in range(true_label_list[i][0]['bboxes'].shape[0]):
                true_grasp_bbox = true_label_list[i][0]['bboxes'][k]                    # xmin, ymin, xmax, ymax
                true_grasp_angle_class = true_label_list[i][0]['labels'][k]                
                true_grasp = get_grasp_from_pred(true_grasp_bbox, true_grasp_angle_class)
                # Angle diff
                angle_diff = np.abs(pred_grasp.as_angle - true_grasp.as_angle) * 180.0 / np.pi
                angle_diff = min(angle_diff, 180.0 - angle_diff)
                angle_diff_list.append(angle_diff)
                #IoU
                bbox_true = true_grasp.as_bbox
                try:
                    p1 = Polygon([bbox_true[0], bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[0]])
                    p2 = Polygon([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[0]])
                    iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
                    iou_list.append(iou)
                except Exception as e: 
                    print('IoU ERROR', e)
                if not correct_pred and angle_diff < 30 and iou > iou_threshold:
                    if not img_correct_pred:
                        img_correct_pred = True
                        correct_img_pred_count += 1
                    correct_pred = True
                    correct_pred_count += 1
                    # Top Score
                    if j == 0:
                        top_score_correct += 1
            # Get indices of incorrect top scorer grasp
            if j == 0 and not img_correct_pred:
                incorrect_top_score.append(i)
    

    # How many images has atleast 1 correct grasp prediction
    grasp_accuracy_img = correct_img_pred_count / len(detections_list)        
    top_score_accuracy_img = top_score_correct / len(detections_list)        

    # How many predicted grasps are actually correct out of all predicted grasps
    if (pred_count == 0):
        grasp_accuracy = 0
    else:
        grasp_accuracy = correct_pred_count / pred_count
        # Avg pred grasp score
        avg_pred_grasp_score /= pred_count

    # Avg IoU
    if len(iou_list) == 0:
        avg_iou = 0
    else:
        avg_iou = sum(iou_list) / len(iou_list)

    # Avg Angle Diff
    if len(angle_diff_list) == 0:
        avg_angle_diff = 0
    else:
        avg_angle_diff = sum(angle_diff_list) / len(angle_diff_list)

    print('Images to check: ', incorrect_top_score)
    print('grasp_acc_img: {:.2f}'.format(grasp_accuracy_img))
    print('top_score_accuracy: {:.4f}'.format(top_score_accuracy_img))
    print('grasp_accuracy: {:.4f}'.format(grasp_accuracy))
    print('avg_iou: {:.2f}'.format(avg_iou))
    print('avg_angle_diff: {:.2f}'.format(avg_angle_diff))
    print('avg_pred_grasp_score: {:.2f}'.format(avg_pred_grasp_score))
    ##################################################################################################    
else:
    ## TEST ON NON LABELLED IMAGES
    # Load list of images
    dataset = '/home/aby/Workspace/Cornell/archive'
    with open(dataset+'/test.txt', 'r') as filehandle: ## To check trained prediction_model on unlabelled unprocessed amazon images
        train_data = json.load(filehandle)
    # Load Image
    for i in range(len(train_data)):
        X = CornellDataset.load_custom_image(dataset+train_data[i])
        disp_img = CornellDataset.load_custom_image(dataset+train_data[i], normalise=False)
        # Expand dim for batch
        X = X[np.newaxis, ...]
        Y_pred = prediction_model.predict(X)

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
