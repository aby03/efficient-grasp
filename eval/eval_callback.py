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
from dataset_processing.grasp import get_grasp_from_pred


def _get_detections(generator,
                    model, 
                    score_threshold=0.05, 
                    max_detections=100,
                    save_path = None):
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

    all_detections = [None  for j in range(generator.size())]
    gt_annotations = [None for j in range(generator.size())]

    # # List of size of total number of images
    # pred_grasps = [None for i in range(generator.size()) ]
    # true_grasps = [None for i in range(generator.size()) ]

    # Run for all images. (DOESNT WORK FOR BATCH SIZE > 1)
    for i in tqdm(range(len(generator))):
        image_bt, true_grasp_bt    = generator.get_annotation_val(i)

        # run network
        boxes, scores, angle_labels = model.predict_on_batch(image_bt)

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]
        
        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]
        
         # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = angle_labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)       

        all_detections[i] = image_detections
        gt_annotations[i] = true_grasp_bt

    return all_detections, gt_annotations
        # raw_image    = generator.load_image(i)
        # image, scale = generator.preprocess_image(raw_image.copy())
        # image, scale = generator.resize_image(image)

        # if keras.backend.image_data_format() == 'channels_first':
        #     image = image.transpose((2, 0, 1))

        # run network
        # pred_grasp_bt = model.predict_on_batch(image_bt)
        # print('........... TEST: ', image_bt.shape, ' V2: ', pred_grasp_bt.shape, ' V3: ', output_bt.shape, ' V4: ', true_grasp_bt.shape)

        # pred_grasps[i*image_bt.shape[0]:(i+1)*image_bt.shape[0]] = pred_grasp_bt
        # true_grasps[i*image_bt.shape[0]:(i+1)*image_bt.shape[0]] = true_grasp_bt

    # print('........... TEST: ', image_bt.shape[0], ' V2: ', len(pred_grasps), ' V3: ', len(true_grasps))
    # return pred_grasps, true_grasps

        # if save_path is not None:
        #     raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        #     draw_annotations(raw_image, generator.load_annotations(i), class_to_bbox_3D = generator.get_bbox_3d_dict(), camera_matrix = generator.load_camera_matrix(i), label_to_name=generator.label_to_name)
        #     draw_detections(raw_image, image_boxes, image_scores, image_labels, image_rotations, image_translations, class_to_bbox_3D = generator.get_bbox_3d_dict(), camera_matrix = generator.load_camera_matrix(i), label_to_name=generator.label_to_name)

        #     cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

def evaluate(
    generator,
    model,
    iou_threshold = 0.25,
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
    all_detections, all_annotations = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    # pred_grasps, true_grasps = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    total_grasp_det = 0
    for p in range(len(all_detections)):
        total_grasp_det += all_detections[p].shape[0]
    print('Total detected grasps: ', total_grasp_det)
    # print('EVAL Det: ', len(all_detections))
    # print('EVAL Det: ', all_detections[0].shape)
    # print('EVAL Annot: ', len(all_annotations))
    # print('EVAL Annot: ', all_annotations[0][0]['bboxes'].shape) # [all_images][0][num_true_grasps_for_image]
    '''
        Multigrasp Model
        pred_grasps: (all_imgs, 100, 7)
        true_grasps: (all_imgs, 30, 6)
    '''
    ## For metrics
    iou_list = []
    angle_diff_list = []
    pred_count = 0
    correct_pred_count = 0      # Total Predictions Count
    correct_img_pred_count = 0  # only 1 prediction counted per image
    avg_pred_grasp_score = 0
    top_score_correct = 0
    # Top k grasps
    top_k = 5
    top_k_correct = 0
    top_k_bool = False
    # For each image in batch
    for i in range(len(all_detections)):
        img_correct_pred = False
        top_k_bool = False
        # For each pred grasp
        for j in range(all_detections[i].shape[0]):
            correct_pred = False
            # Create predicted grasp in right format
            pred_grasp_bbox = all_detections[i][j][0:4]    # xmin, ymin, xmax, ymax
            pred_grasp_score = all_detections[i][j][4]
            pred_grasp_angle_class = all_detections[i][j][5]
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
            for k in range(all_annotations[i][0]['bboxes'].shape[0]):
                true_grasp_bbox = all_annotations[i][0]['bboxes'][k]                    # xmin, ymin, xmax, ymax
                true_grasp_angle_class = all_annotations[i][0]['labels'][k]                
                true_grasp = get_grasp_from_pred(true_grasp_bbox, true_grasp_angle_class)
                # Angle diff
                angle_diff = np.abs(pred_grasp.as_angle - true_grasp.as_angle) * 180.0 / np.pi
                angle_diff = min(angle_diff, 180.0 - angle_diff)
                min_angle_diff = min(min_angle_diff, angle_diff)
                #IoU
                bbox_true = true_grasp.as_bbox
                try:
                    p1 = Polygon([bbox_true[0], bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[0]])
                    p2 = Polygon([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[0]])
                    iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
                    max_iou = max(max_iou, iou)
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
                    # Top k score
                    if j < top_k and not top_k_bool:
                        top_k_bool = True
                        top_k_correct += 1
            
            angle_diff_list.append(min_angle_diff)
            iou_list.append(max_iou)
            
    
 
    # How many images has atleast 1 correct grasp prediction
    grasp_accuracy_img = correct_img_pred_count / len(all_detections)        
    # Accuracy based on top scorer grasp
    top_score_accuracy_img = top_score_correct / len(all_detections)
    # Accuracy based on top k grasps
    top_k_acc_img = top_k_correct / len(all_detections)        

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
        avg_angle_diff = 90
    else:
        avg_angle_diff = sum(angle_diff_list) / len(angle_diff_list)

    return grasp_accuracy_img, top_score_accuracy_img, top_k_acc_img, grasp_accuracy, avg_iou, avg_angle_diff, avg_pred_grasp_score

    # ## IoU Angle Diff
    # correct_grasp_count = 0
    # iou_list = []
    # angle_diff_list = []
    # ### TO DO FROM HERE
    # ## For Image
    # for i in range(len(pred_grasps)):
    #     correct_image_pred = False
    #     iou_list_img = []
    #     angle_diff_list_img = []

    #     ## For Each Pred Grasp
    #     for j in range(len(pred_grasps[i])):
    #         index = min_loss_index[i][j]
    #         # Converted to Grasp obj, unnormalized, in [y,x] format
    #         true_grasp_obj = Grasp(true_grasps[i][index][0:2], *true_grasps[i][index][2:], unnorm = True)
    #         pred_grasp_obj = Grasp(pred_grasps[i][j][0:2], *pred_grasps[i][j][2:6], unnorm=True)
    #         # converted to list of bboxes in [x, y] format
    #         bbox_true = true_grasp_obj.as_bbox
    #         bbox_pred = pred_grasp_obj.as_bbox
            
    #         #IoU
    #         try:
    #             p1 = Polygon([bbox_true[0], bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[0]])
    #             p2 = Polygon([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[0]])
    #             iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
    #             iou_list_img.append(iou)
    #         except Exception as e: 
    #             print('IoU ERROR', e)
    #             print('Bbox pred:', bbox_pred)
    #             print('pred grasp:', pred_grasps[j])
    #             print('Bbox true:', bbox_true)
            
    #         #Angle Diff
    #         true_sin = true_grasp_obj.sin_t
    #         true_cos = true_grasp_obj.cos_t  
    #         if true_cos != 0:
    #             true_angle = np.arctan(true_sin/true_cos) * 180/np.pi
    #         else:
    #             true_angle = 90
            
    #         pred_sin = pred_grasp_obj.sin_t
    #         pred_cos = pred_grasp_obj.cos_t
    #         if pred_cos != 0:
    #             pred_angle = np.arctan(pred_sin/pred_cos) * 180/np.pi
    #         else:
    #             pred_angle = 90
    #         # true_angle = true_grasps[j][index][2] * 180.0/np.pi
    #         # pred_angle = pred_grasps[j][2] * 180.0/np.pi
            
    #         angle_diff = np.abs(pred_angle - true_angle)
    #         angle_diff = min(angle_diff, 180.0 - angle_diff)
    #         angle_diff_list_img.append(angle_diff)
            
    #         if angle_diff < 30. and iou >= 0.25:
    #             correct_image_pred = True
    #             # print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
    #     iou_list.append(iou_list_img)
    #     angle_diff_list.append(angle_diff_list_img)
    #     if correct_image_pred:
    #         correct_grasp_count += 1

    # grasp_accuracy = correct_grasp_count / len(pred_grasps)
    # avg_iou = sum([sum(iou_v) for iou_v in iou_list]) / (len(pred_grasps)*len(pred_grasps[0]))
    # avg_angle_diff = sum([sum(ang_v) for ang_v in angle_diff_list]) / (len(pred_grasps)*len(pred_grasps[0]))
    # ### TO DO END HERE

    # return avg_grasp_loss, grasp_accuracy, avg_iou, avg_angle_diff

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
        grasp_accuracy_img, top_score_accuracy_img, top_k_acc_img, grasp_accuracy, avg_iou, avg_angle_diff, avg_pred_grasp_score = evaluate(
            self.generator,
            self.active_model,
            iou_threshold=self.iou_threshold, # start
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            diameter_threshold = self.diameter_threshold, # end
            save_path=self.save_path,
            multi_pred=self.multi_pred
        )
        
        if self.tensorboard is not None:
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer is not None:
                summary = tf.Summary()
                # Grasp Loss
                summary_value_img_grasp_acc = summary.value.add()
                summary_value_img_grasp_acc.simple_value = grasp_accuracy_img
                summary_value_img_grasp_acc.tag = "grasp_acc_img"
                # Grasp Top score accuracy
                summary_value_top_score_acc = summary.value.add()
                summary_value_top_score_acc.simple_value = top_score_accuracy_img
                summary_value_top_score_acc.tag = "top_score_accuracy_img"
                # Grasp Top k grasps by score accuracy
                summary_value_top_k_acc_img = summary.value.add()
                summary_value_top_k_acc_img.simple_value = top_k_acc_img
                summary_value_top_k_acc_img.tag = "top_k_acc_img"
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
                # Average pred grasp score
                summary_value_avg_pred_grasp_score = summary.value.add()
                summary_value_avg_pred_grasp_score.simple_value = avg_pred_grasp_score
                summary_value_avg_pred_grasp_score.tag = "avg_pred_grasp_score"
                self.tensorboard.writer.add_summary(summary, epoch)
            else:
                with self.summary_writer.as_default():
                    tf.summary.scalar('grasp_acc_img', grasp_accuracy_img, step=epoch)
                    tf.summary.scalar('top_score_accuracy_img', top_score_accuracy_img, step=epoch)
                    tf.summary.scalar('top_k_accuracy', top_k_acc_img, step=epoch)
                    tf.summary.scalar('grasp_accuracy', grasp_accuracy, step=epoch)
                    tf.summary.scalar('avg_iou', avg_iou, step=epoch)
                    tf.summary.scalar('avg_angle_diff', avg_angle_diff, step=epoch)
                    tf.summary.scalar('avg_pred_grasp_score', avg_pred_grasp_score, step=epoch)
                    self.summary_writer.flush()

        logs['grasp_acc_img'] = grasp_accuracy_img
        logs['top_score_accuracy_img'] = top_score_accuracy_img
        logs['top_k_accuracy'] = top_k_acc_img
        logs['grasp_accuracy'] = grasp_accuracy
        logs['avg_iou'] = avg_iou
        logs['avg_angle_diff'] = avg_angle_diff
        logs['avg_pred_grasp_score'] = avg_pred_grasp_score

        if self.verbose == 1:
            print('grasp_acc_img: {:.2f}'.format(grasp_accuracy_img))
            print('top_score_accuracy_img: {:.2f}'.format(top_score_accuracy_img))
            print('top_k_accuracy: {:.2f}'.format(top_k_acc_img))
            print('grasp_accuracy: {:.4f}'.format(grasp_accuracy))
            print('avg_iou: {:.2f}'.format(avg_iou))
            print('avg_angle_diff: {:.2f}'.format(avg_angle_diff))
            print('avg_pred_grasp_score: {:.2f}'.format(avg_pred_grasp_score))
        return logs


                    

