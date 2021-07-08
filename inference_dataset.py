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

### MAIN
# prediction_model.load_weights('checkpoints/2021_06_24_02_57_59/vmrd_best_val_loss.h5', by_name=True) ##VMRD
# prediction_model.load_weights('checkpoints/2021_06_23_20_33_49/cornell_finish.h5', by_name=True) ##CORNELL-RGD
# prediction_model.load_weights('checkpoints/2021_06_25_18_19_28/cornell_finish.h5', by_name=True) ##CORNELL-RGB N3b 2021_06_25_18_19_28
# prediction_model.load_weights('checkpoints/2021_06_27_22_45_12/amazon_best_val_loss.h5', by_name=True) ##Amazon
# prediction_model.load_weights('checkpoints/2021_06_27_22_45_12/amazon_finish.h5', by_name=True) ##Amazon 

# Different Model
# prediction_model.load_weights('checkpoints/2021_06_28_01_56_08/cornell_finish.h5', by_name=True) ##Cornell Light 2021_06_28_01_56_08

# prediction_model.load_weights('checkpoints/2021_06_30_19_01_35/vmrd_best_val_loss.h5', by_name=True) ##VMRD

### New Setup
prediction_model.load_weights('checkpoints/2021_07_08_06_34_47/cornell_best_val_loss.h5', by_name=True)

print("Weights loaded!")

dataset_name = "cornell" # Values can be [cornell, vmrd, amazon]
RGD_DATA = True
run_dataset = True
SAVE_FIGURE = True
### to show all keep Show_plots true and show_selective_plots false
### to show select keep both true
SHOW_PLOTS = True
SHOW_SELECTIVE_PLOTS = False
# selective_plots = [70, 80, 92, 97, 113, 118, 132]  # For Cornell-RGD N3
selective_plots = [50, 59, 72, 84, 89, 105, 113, 118, 148, 162] # For Cornell-RGB N3b
# FOr VMRD BELOW
# selective_plots = [1, 6, 8, 9, 20, 21, 25, 27, 40, 48, 50, 56, 59, 61, 65, 69, 71, 73, 75, 76, 77, 80, 81, 82, 89, 91, 92, 97, 105, 110, 115, 116, 118, 119, 127, 131, 133, 140, 142, 157, 159, 160, 161, 165, 169, 172, 177, 181, 192, 195, 196, 200, 217, 218, 222, 223, 232, 245, 247, 251, 252, 268, 304, 312, 325, 326, 353, 364, 365, 366, 369, 388, 408, 414, 424, 427, 429, 431, 436, 443, 467, 474, 476, 511, 572, 614, 633, 657, 658, 669, 698, 745, 818, 821, 851, 853, 854, 906, 916, 946, 956, 957, 969, 971, 1048, 1075, 1088, 1092, 1095, 1106, 1107, 1112, 1114, 1118, 1123, 1128, 1129, 1133, 1142, 1145, 1147, 1149, 1154, 1161, 1172, 1175, 1190, 1198, 1200, 1201, 1202, 1204, 1212, 1235, 1236, 1239, 1240, 1241, 1245, 1246, 1252, 1257, 1259, 1264, 1265, 1270, 1272, 1277, 1281, 1283, 1284, 1287, 1293, 1294, 1296, 1299, 1302, 1305, 1308, 1309, 1316, 1319, 1320, 1324, 1328, 1336, 1339, 1341, 1342, 1345, 1349, 1356, 1361, 1368, 1376, 1381, 1384, 1390, 1392, 1393, 1396, 1398, 1400, 1401, 1406, 1416, 1417, 1419, 1421, 1425, 1426, 1437, 1440, 1447, 1459, 1460, 1461, 1462, 1471, 1480, 1491, 1494, 1519, 1534, 1541, 1547, 1548, 1559, 1566, 1567, 1569, 1571, 1577, 1580, 1582, 1583, 1584, 1586, 1595, 1606, 1608, 1609, 1614, 1615, 1622, 1624, 1629, 1630, 1634, 1640, 1643, 1652, 1657, 1661, 1670, 1672, 1675, 1678, 1679, 1680, 1682, 1684, 1685, 1691, 1693, 1694, 1702, 1703, 1704, 1708, 1717, 1727, 1747, 1774, 1779, 1788, 1793, 1807, 1817, 1819, 1831, 1837, 1855, 1869, 1870, 1871, 1872, 1876, 1878, 1880, 1948, 1959, 1966, 1975, 1988, 1993, 1997, 2010, 2012, 2030, 2036, 2046, 2056, 2073, 2088, 2100, 2104, 2126, 2127, 2128, 2132, 2135, 2136, 2144, 2145, 2146, 2147, 2158, 2159, 2168, 2173, 2178, 2179, 2182, 2188, 2190, 2194, 2201, 2202, 2203, 2214, 2244, 2262, 2280, 2281, 2286, 2292, 2294, 2297, 2298, 2299, 2302, 2304, 2305, 2306, 2307, 2311, 2335, 2348, 2350, 2351, 2357, 2360, 2361, 2362, 2367, 2378, 2383, 2386, 2401, 2416, 2420, 2425, 2441, 2443, 2449, 2462, 2474, 2506, 2507, 2518, 2519, 2558, 2607, 2610, 2612, 2643, 2650, 2652, 2656, 2661, 2666, 2681, 2692, 2695, 2696, 2712, 2736, 2741, 2743, 2744, 2746, 2764, 2769, 2775, 2783, 2822, 2824, 2825, 2826, 2827, 2837, 2839, 2847, 2855, 2859, 2860, 2862, 2865, 2866, 2868, 2871, 2873, 2874, 2875, 2888, 2893, 2894, 2895, 2896, 2900, 2912, 2916, 2917, 2921, 2927, 2929, 2937, 2938, 2947, 2952, 2961, 2964, 2969, 2972, 2977, 2981, 2985, 2988, 2989, 2993, 3000, 3002, 3006, 3013, 3014, 3017, 3018, 3022, 3024, 3025, 3027, 3035, 3040, 3058, 3074, 3094, 3095, 3101, 3106, 3107, 3111, 3115, 3127, 3133, 3166, 3169, 3170, 3181, 3189, 3203, 3226, 3229, 3249, 3276, 3284, 3287, 3322, 3325, 3337, 3338, 3343, 3355, 3357, 3362, 3364, 3367, 3383, 3386, 3392, 3393, 3411, 3421, 3434, 3445, 3461, 3465, 3470, 3472, 3481, 3487, 3495, 3510, 3521, 3559]

if run_dataset:
    if dataset_name == "cornell":
        # Load list of images
        dataset = '/home/aby03/Workspace/MTP/Datasets/Cornell/archive'
        # with open(dataset+'/train_overfit_2.txt', 'r') as filehandle:
        with open(dataset+'/valid_1.txt', 'r') as filehandle:
        # with open(dataset+'/amazon_test.txt', 'r') as filehandle: ## To check Cornell trained model on amazon images
            train_data = json.load(filehandle)
        
        val_generator = CornellDataset(
            dataset,
            train_data,
            train=False,
            shuffle=False,
            batch_size=1,
            rgd_mode=RGD_DATA
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
            if RGD_DATA:
                rgd_img = image.Image.from_file(dataset+train_data[i])
            else:
                rgd_img = image.Image.from_file(dataset+train_data[i].replace('z.png', 'r.png'))
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

        if SHOW_PLOTS and all_score.shape[0] > 0:
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
            ax[0][1].imshow(disp_img)
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
    top_k = 5
    top_k_correct = 0
    top_k_bool = False
    incorrect_top_score = [] # To find images with bad top score grasp
    precision_count = 0
    # For each image in batch
    for i in range(len(detections_list)):
        img_correct_pred = False
        top_k_bool = False
        # For each pred grasp
        # for j in range(min(detections_list[i].shape[0], top_k)):
        for j in range(detections_list[i].shape[0]):
            correct_pred = False
            precision_bool = False
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

            min_angle_diff = 90
            max_iou = 0.0
            # For each true_grasp
            for k in range(true_label_list[i][0]['bboxes'].shape[0]):
                true_grasp_bbox = true_label_list[i][0]['bboxes'][k]                    # xmin, ymin, xmax, ymax
                true_grasp_angle_class = true_label_list[i][0]['labels'][k]                
                true_grasp = get_grasp_from_pred(true_grasp_bbox, true_grasp_angle_class)
                # Angle diff
                angle_diff = np.abs(pred_grasp.as_angle - true_grasp.as_angle) * 180.0 / np.pi
                angle_diff = min(angle_diff, 180.0 - angle_diff)
                #IoU
                bbox_true = true_grasp.as_bbox
                try:
                    p1 = Polygon([bbox_true[0], bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[0]])
                    p2 = Polygon([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[0]])
                    iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
                except Exception as e: 
                    print('IoU ERROR', e)
                max_iou = max(max_iou, iou)
                if iou > 0.25:
                    min_angle_diff = min(min_angle_diff, angle_diff)
                if iou > 0.5 and not precision_bool:
                    precision_bool = True
                    precision_count += 1
                if not correct_pred and angle_diff < 30 and iou > iou_threshold:
                    if not img_correct_pred:
                        img_correct_pred = True
                        correct_img_pred_count += 1
                    correct_pred = True
                    correct_pred_count += 1
                    # Top Score
                    if j == 0:
                        top_score_correct += 1
                    # Top k score
                    if j < top_k and not top_k_bool:
                        top_k_bool = True
                        top_k_correct += 1
            
            # Get angle diff and IoU for the pred grasp
            angle_diff_list.append(min_angle_diff)
            iou_list.append(max_iou)
            
            # Get indices of incorrect top scorer grasp
            if j == 0 and not img_correct_pred:
                incorrect_top_score.append(i)
    

    # How many images has atleast 1 correct grasp prediction
    grasp_accuracy_img = correct_img_pred_count / len(detections_list)        
    top_score_accuracy_img = top_score_correct / len(detections_list)        
    top_k_acc_img = top_k_correct / len(detections_list)        

    # How many predicted grasps are actually correct out of all predicted grasps
    if (pred_count == 0):
        grasp_accuracy = 0
        precision = 0
    else:
        grasp_accuracy = correct_pred_count / pred_count
        precision = precision_count / pred_count
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
    print('grasp_acc_img: {:.4f}'.format(grasp_accuracy_img))
    print('top_score_accuracy: {:.4f}'.format(top_score_accuracy_img)) 
    print('top_k_accuracy: {:.4f} for k = {}'.format(top_k_acc_img, top_k)) 
    print('Precision: {:.4f}'.format(precision))
    print('grasp_accuracy: {:.4f}'.format(grasp_accuracy))
    print('avg_iou: {:.4f}'.format(avg_iou))
    print('avg_angle_diff: {:.4f}'.format(avg_angle_diff))
    print('avg_pred_grasp_score: {:.4f}'.format(avg_pred_grasp_score))
    ##################################################################################################    
else:
    ## TEST ON NON LABELLED IMAGES
    # Load list of images
    dataset = '/home/aby/Workspace/custom-test-images'
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

        # Plots
        DISPLAY_PRED=100
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(10, 4)
        ## Show Original Image
        ax[0].imshow(disp_img)
        ax[0].set_title('Original Image')
        ## Show RGD image and set axis title
        ax[1].imshow(disp_img)
        ax[1].set_title('Predicted Grasps on Image')
        # Show Predicted grasps on orig image
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




        # # Remove batch dim
        # Y_out = Y_pred[0,:,:]

        # all_predictions = Y_out[:,0:6]
        # all_score = Y_out[:,6]
        
        # DISPLAY_PRED = 10
        # # Sort y_out based on score
        # sort_index = (-all_score).argsort()
        # all_score = all_score[sort_index]
        # all_predictions = all_predictions[sort_index,:]
        # print(all_score[:DISPLAY_PRED])

        # ### Plot all grasps
        # fig, ax = plt.subplots(1,2)
        # fig.set_size_inches(10, 5)
        # ## Show Original Image
        # ax[0].imshow(disp_img)
        # ax[0].set_title('Original Image')
        # ## Show RGD image and set axis title
        # ax[1].imshow(disp_img)
        # ax[1].set_title('Predicted Grasps')
        # ## Show Predicted grasps on orig image
        # colormap=plt.get_cmap('cool')
        # display_score = all_score[:DISPLAY_PRED]
        # center_y = []
        # center_x = []
        # col_norm = matplotlib.colors.Normalize(vmin=display_score[-1], vmax=display_score[0], clip=False)
        # for j in range(min(all_predictions.shape[0], DISPLAY_PRED)):
        #     plot_grasp = Grasp(all_predictions[j][0:2], *all_predictions[j][2:], quality=display_score[j], unnorm=True)
        #     plot_points = plot_grasp.as_gr.points
        #     points = np.vstack((plot_points, plot_points[0]))
        #     ax[1].plot(points[:, 1], points[:, 0], color=colormap(col_norm(display_score[j])))
        #     center_y.append(plot_grasp.center[0])
        #     center_x.append(plot_grasp.center[1])
        # sc=ax[1].scatter(center_x, center_y, c=display_score, cmap=colormap)
        # # ax[1][1].scatter(center_x, center_y, c=display_score, cmap=colormap)
        # plt.colorbar(sc)
        # if SAVE_FIGURE:
        #     fname = '/home/aby/Pictures/'+str(i)
        #     plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        #         orientation='portrait', papertype=None, format=None,
        #         transparent=False, bbox_inches='tight', pad_inches=0.1,
        #         frameon=None, metadata=None)
        # plt.show()

        # # Plot predicted grasp
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        # ax.imshow(disp_img)
        # pred_grasp.plot(ax, 'red')
        # plt.show()
