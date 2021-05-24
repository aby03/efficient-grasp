import tensorflow as tf
from tensorflow import keras
# import progressbar
from tqdm import tqdm
import numpy as np
# from eval.common import evaluate
import math
import sys
sys.path.append(".")
# from generators.cornell import *
from dataset_processing.grasp import Grasp
from losses import grasp_loss

from shapely import speedups
speedups.disable()
from shapely.geometry import Polygon # For IoU


def _get_detections(generator, model, save_path = None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = (boxes+classes = detections[num_detections, 4 + num_classes], rotations = detections[num_detections, num_rotation_parameters], translations = detections[num_detections, num_translation_parameters)

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """

    # all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    pred_grasps = [None for i in range(generator.size()) ]
    true_grasps = [None for i in range(generator.size()) ]

    for i in tqdm(range(generator.size())):
        image_bt, output_bt    = generator[i]
        true_grasp_bt = output_bt
        # raw_image    = generator.load_image(i)
        # image, scale = generator.preprocess_image(raw_image.copy())
        # image, scale = generator.resize_image(image)

        # if keras.backend.image_data_format() == 'channels_first':
        #     image = image.transpose((2, 0, 1))

        # run network
        pred_grasp_bt = model.predict_on_batch(image_bt)

        pred_grasps[i*image_bt.shape[0]:(i+1)*image_bt.shape[0]] = pred_grasp_bt
        true_grasps[i*image_bt.shape[0]:(i+1)*image_bt.shape[0]] = true_grasp_bt

    return pred_grasps, true_grasps

        # if save_path is not None:
        #     raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        #     draw_annotations(raw_image, generator.load_annotations(i), class_to_bbox_3D = generator.get_bbox_3d_dict(), camera_matrix = generator.load_camera_matrix(i), label_to_name=generator.label_to_name)
        #     draw_detections(raw_image, image_boxes, image_scores, image_labels, image_rotations, image_translations, class_to_bbox_3D = generator.get_bbox_3d_dict(), camera_matrix = generator.load_camera_matrix(i), label_to_name=generator.label_to_name)

        #     cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

def evaluate(
    generator,
    model,
    iou_threshold = 0.5,
    score_threshold = 0.05,
    max_detections = 100,
    save_path = None,
    diameter_threshold = 0.1,
    multi_pred=False
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save images with visualized detections to.
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
    # Returns
        Several dictionaries mapping class names to the computed metrics.
    """
    # gather all detections and annotations
    pred_grasps, true_grasps = _get_detections(generator, model, save_path=save_path)
    
    if not multi_pred:
        ## Single Grasp Model
        # Grasp Loss
        loss_v = []
        min_loss_index = []
        # For an image
        for j in range(len(true_grasps)):
            min_loss = float('inf')
            min_index = 0
            # For a grasp
            for i in range(len(true_grasps[j])):
                cur_loss = grasp_loss(true_grasps[j][i], pred_grasps[j])
                # print('Im {}, G {}, Loss {}'.format(j, i, cur_loss))
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    min_index = i
            loss_v.append(min_loss)
            min_loss_index.append(min_index)
            # print(' === Im {} Min Loss {} Loss Index {}'.format(j, min_loss, min_index))
        avg_grasp_loss = sum(loss_v) / len(loss_v)

        # IoU Angle Diff
        correct_grasp_count = 0
        iou_list = []
        angle_diff_list = []
        for j in range(len(true_grasps)):
            index = min_loss_index[j]
            # Converted to Grasp obj, unnormalized, in [y,x] format
            true_grasp_obj = Grasp(true_grasps[j][index][0:2], *true_grasps[j][index][2:], unnorm = True)
            pred_grasp_obj = Grasp(pred_grasps[j][0:2], *pred_grasps[j][2:], unnorm=True)
            # converted to list of bboxes in [x, y] format
            bbox_true = true_grasp_obj.as_bbox
            bbox_pred = pred_grasp_obj.as_bbox
            
            #IoU
            try:
                p1 = Polygon([bbox_true[0], bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[0]])
                p2 = Polygon([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[0]])
                iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
                iou_list.append(iou)
            except Exception as e: 
                print('IoU ERROR', e)
                print('Bbox pred:', bbox_pred)
                print('pred grasp:', pred_grasps[j])
                print('Bbox true:', bbox_true)
            
            #Angle Diff
            true_sin = true_grasp_obj.sin_t
            true_cos = true_grasp_obj.cos_t  
            if true_cos != 0:
                true_angle = np.arctan(true_sin/true_cos) * 180/np.pi
            else:
                true_angle = 90
            
            pred_sin = pred_grasp_obj.sin_t
            pred_cos = pred_grasp_obj.cos_t
            if pred_cos != 0:
                pred_angle = np.arctan(pred_sin/pred_cos) * 180/np.pi
            else:
                pred_angle = 90
            # true_angle = true_grasps[j][index][2] * 180.0/np.pi
            # pred_angle = pred_grasps[j][2] * 180.0/np.pi
            
            angle_diff = np.abs(pred_angle - true_angle)
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            angle_diff_list.append(angle_diff)
            
            if angle_diff < 30. and iou >= 0.25:
                correct_grasp_count += 1
                # print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
        grasp_accuracy = correct_grasp_count / len(true_grasps)
        avg_iou = sum(iou_list) / len(true_grasps)
        avg_angle_diff = sum(angle_diff_list) / len(true_grasps)
    else:
        '''
            Multigrasp Model
            pred_grasps: (b, 100, 7)
            true_grasps: (b, 30, 6)
        '''
        ## Grasp Loss
        loss_v = []
        min_loss_index = []
        # For each image
        for i in range(len(pred_grasps)):
            loss_grasp = []
            min_index_grasp = []
            # For each pred grasp
            for j in range(len(pred_grasps[i])):
                min_loss = float('inf')
                min_index = 0
                # For each true_grasp
                for k in range(len(true_grasps[i])):
                    cur_loss = grasp_loss(true_grasps[i][k], pred_grasps[i][j][0:6])
                    # print('Im {}, G {}, Loss {}'.format(j, i, cur_loss))
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        min_index = k
                loss_grasp.append(min_loss)
                min_index_grasp.append(min_index)
            loss_v.append(loss_grasp)
            min_loss_index.append(min_index_grasp)
        avg_grasp_loss = sum([sum(los) for los in loss_v]) / len(loss_v)*len(loss_v[0])
        
        ## IoU Angle Diff
        correct_grasp_count = 0
        iou_list = []
        angle_diff_list = []
        ### TO DO FROM HERE
        ## For Image
        for i in range(len(pred_grasps)):
            ## For Each Pred Grasp
            for j in range(len(pred_grasps[i])):
                index = min_loss_index[i][j]
                # Converted to Grasp obj, unnormalized, in [y,x] format
                true_grasp_obj = Grasp(true_grasps[i][index][0:2], *true_grasps[i][index][2:], unnorm = True)
                pred_grasp_obj = Grasp(pred_grasps[i][j][0:2], *pred_grasps[i][j][2:6], unnorm=True)
                # converted to list of bboxes in [x, y] format
                bbox_true = true_grasp_obj.as_bbox
                bbox_pred = pred_grasp_obj.as_bbox
                
                #IoU
                try:
                    p1 = Polygon([bbox_true[0], bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[0]])
                    p2 = Polygon([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[0]])
                    iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
                    iou_list.append(iou)
                except Exception as e: 
                    print('IoU ERROR', e)
                    print('Bbox pred:', bbox_pred)
                    print('pred grasp:', pred_grasps[j])
                    print('Bbox true:', bbox_true)
                
                #Angle Diff
                true_sin = true_grasp_obj.sin_t
                true_cos = true_grasp_obj.cos_t  
                if true_cos != 0:
                    true_angle = np.arctan(true_sin/true_cos) * 180/np.pi
                else:
                    true_angle = 90
                
                pred_sin = pred_grasp_obj.sin_t
                pred_cos = pred_grasp_obj.cos_t
                if pred_cos != 0:
                    pred_angle = np.arctan(pred_sin/pred_cos) * 180/np.pi
                else:
                    pred_angle = 90
                # true_angle = true_grasps[j][index][2] * 180.0/np.pi
                # pred_angle = pred_grasps[j][2] * 180.0/np.pi
                
                angle_diff = np.abs(pred_angle - true_angle)
                angle_diff = min(angle_diff, 180.0 - angle_diff)
                angle_diff_list.append(angle_diff)
                
                if angle_diff < 30. and iou >= 0.25:
                    correct_grasp_count += 1
                    # print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
        grasp_accuracy = correct_grasp_count / len(pred_grasps)*len(pred_grasps[0])
        avg_iou = sum(iou_list) / len(pred_grasps)*len(pred_grasps[0])
        avg_angle_diff = sum(angle_diff_list) / len(pred_grasps)*len(pred_grasps[0])
        ### TO DO END HERE

    return avg_grasp_loss, grasp_accuracy, avg_iou, avg_angle_diff

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        model,
        iou_threshold = 0.5,
        score_threshold = 0.05,
        max_detections = 100,
        diameter_threshold = 0.1,
        save_path = None,
        tensorboard = None,
        weighted_average = False,
        verbose = 1,
        multi_pred = False
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator: The generator that represents the dataset to evaluate.
            model: The model to evaluate.
            iou_threshold: The threshold used to consider when a detection is positive or negative.
            score_threshold: The score confidence threshold to use for detections.
            max_detections: The maximum number of detections to use per image.
            diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
            save_path: The path to save images with visualized detections to.
            tensorboard: Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average: Compute the mAP using the weighted average of precisions among classes.
            verbose: Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose
        self.diameter_threshold = diameter_threshold
        self.active_model = model
        self.multi_pred = multi_pred
        self.summary_writer = tf.summary.create_file_writer(self.tensorboard.log_dir+"/validation")


        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs):
        # logs = logs or {}

        # run evaluation
        avg_grasp_loss, grasp_accuracy, avg_iou, avg_angle_diff = evaluate(
            self.generator,
            self.active_model,
            save_path=self.save_path,
            multi_pred=self.multi_pred
        )
        
        if self.tensorboard is not None:
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer is not None:
                summary = tf.Summary()
                # Grasp Loss
                summary_value_avg_grasp_loss = summary.value.add()
                summary_value_avg_grasp_loss.simple_value = avg_grasp_loss
                summary_value_avg_grasp_loss.tag = "val_grasp_loss"
                # Grasp Accuracy
                summary_value_grasp_accuracy = summary.value.add()
                summary_value_grasp_accuracy.simple_value = grasp_accuracy
                summary_value_grasp_accuracy.tag = "grasp_accuracy"
                # Average IoU
                summary_value_avg_iou = summary.value.add()
                summary_value_avg_iou.simple_value = avg_iou
                summary_value_avg_iou.tag = "avg_iou"
                # Average Angle Diff
                summary_value_avg_angle_diff = summary.value.add()
                summary_value_avg_angle_diff.simple_value = avg_angle_diff
                summary_value_avg_angle_diff.tag = "avg_angle_diff"

                self.tensorboard.writer.add_summary(summary, epoch)
            else:
                with self.summary_writer.as_default():
                    tf.summary.scalar('val_grasp_loss', avg_grasp_loss, step=epoch)
                    tf.summary.scalar('grasp_accuracy', grasp_accuracy, step=epoch)
                    tf.summary.scalar('avg_iou', avg_iou, step=epoch)
                    tf.summary.scalar('avg_angle_diff', avg_angle_diff, step=epoch)
                    self.summary_writer.flush()

        logs['val_grasp_loss'] = avg_grasp_loss
        logs['grasp_accuracy'] = grasp_accuracy
        logs['avg_iou'] = avg_iou
        logs['avg_angle_diff'] = avg_angle_diff

        if self.verbose == 1:
            print('val_grasp_loss: {:.2f}'.format(avg_grasp_loss))
            print('grasp_accuracy: {:.4f}'.format(grasp_accuracy))
            print('avg_iou: {:.2f}'.format(avg_iou))
            print('avg_angle_diff: {:.2f}'.format(avg_angle_diff))
        return logs
