
# python train_multi.py --phi 0 --batch-size 1 --lr 1e-4 --epochs 200 --no-snapshots --weights imagenet cornell /kaggle/input/cornell-preprocessed/Cornell/archive

# Starting training timer
from datetime import datetime
start_time = datetime.now()

import argparse
import sys
import time
import os
import json
# Preprocessing
from dataset_processing.cornell_generator import CornellDataset
from dataset_processing.amazon_generator import AmazonDataset
from dataset_processing.vmrd_generator import VMRDDataset

import tensorflow as tf
## To supress console output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Optimization after profiling
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Model related
from model_multi import build_EfficientGrasp_multi
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from losses import focal, smooth_l1, grasp_loss_multi
from custom_load_weights import custom_load_weights
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

def parse_args(args):
    """
    Parse the arguments.
    """
    date_and_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    parser = argparse.ArgumentParser(description = 'EfficientGrasp training script.')

    subparsers = parser.add_subparsers(help = 'Arguments for specific dataset types.', dest = 'dataset_type')
    subparsers.required = True
    cornell_parser = subparsers.add_parser('cornell')
    cornell_parser.add_argument('cornell_path', help = 'Path to dataset directory (ie. Datasets/Cornell/archive/).')

    amazon_parser = subparsers.add_parser('amazon')
    amazon_parser.add_argument('amazon_path', help = 'Path to dataset directory (ie. Datasets/Cornell/archive/).')

    vmrd_parser = subparsers.add_parser('vmrd')
    vmrd_parser.add_argument('vmrd_path', help = 'Path to dataset directory (ie. Datasets/Cornell/archive/).')

    parser.add_argument('--weights', help = 'File containing weights to init the model parameter')
    parser.add_argument('--freeze-backbone', help = 'Freeze training of backbone layers.', action = 'store_true')

    parser.add_argument('--batch-size', help = 'Size of the batches.', default = 1, type = int)
    parser.add_argument('--lr', help = 'Learning rate', default = 1e-4, type = float)
    parser.add_argument('--phi', help = 'Hyper parameter phi', default = 0, type = int, choices = (0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help = 'Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs', help = 'Number of epochs to train.', type = int, default = 100)
    parser.add_argument('--start-epoch', help = 'Epoch count to start for resuming training', dest = 'start_epoch', type = int, default = 0)
    parser.add_argument('--steps', help = 'Number of steps per epoch.', type = int, default = int(179 * 10))
    parser.add_argument('--snapshot-path', help = 'Path to store snapshots of models during training', default = os.path.join("checkpoints", date_and_time))
    parser.add_argument('--tensorboard-dir', help = 'Log directory for Tensorboard output', default = os.path.join("logs", date_and_time))
    parser.add_argument('--no-snapshots', help = 'Disable saving snapshots.', dest = 'snapshots', action = 'store_false')
    parser.add_argument('--no-evaluation', help = 'Disable per epoch evaluation.', dest = 'evaluation', action = 'store_false')

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help = 'Use multiprocessing in fit_generator.', action = 'store_true')
    parser.add_argument('--workers', help = 'Number of generator workers.', type = int, default = 1)
    parser.add_argument('--max-queue-size', help = 'Queue length for multiprocessing workers in fit_generator.', type = int, default = 1)
    
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)

def main(args = None):
    """
    Train an EfficientGrasp model.

    Args:
        args: parseargs object containing configuration for the training procedure.
    """
    # Fix cudnn initialization error for tf 1.15 py3.7 CUDA 10.2 EfficientGrasp environment 
    # import tensorflow as tf
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # import tensorflow as tf
    # physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generators
    print("\nCreating the Generators...\n")
    train_generator, validation_generator = create_generators(args)
    print("Done!")

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("\nBuilding Model!\n")
    model, prediction_model, all_layers = build_EfficientGrasp_multi(args.phi,
                                num_classes=12,
                                print_architecture=False)

    # load pretrained weights
    if args.weights:
        if args.weights == 'imagenet':
            model_name = 'efficientnet-b{}'.format(args.phi)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
        else:
            print('Loading model, this may take a second...')
            custom_load_weights(filepath = args.weights, layers = all_layers, skip_mismatch = True)
            print("\nCustom Weights Loaded!\n")

    # freeze backbone layers
    if args.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi]):
            model.layers[i].trainable = False

    print("\nCompiling Model!\n")
    model.compile(  
                    optimizer=Adam(lr = args.lr, clipnorm = 0.001),
                    loss={'regression_out': grasp_loss_multi(args.batch_size),
                        #   'regression_score' score_loss_multi(),
                         }
                    # metric=['grasp_accuracy']
                 )
    model.compile(optimizer=Adam(lr = args.lr, clipnorm = 0.001), 
                  loss={'bbox_regression': smooth_l1(),
                        'angle_classification': focal(),
                        },
                  loss_weights = {'bbox_regression' : 2.0,
                                  'angle_classification': 1.0
                                 })

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )
    
    # ## TEST ON SINGLE IMAGE
    # import numpy as np
    # # filename = '/kaggle/input/cornell-preprocessed/Cornell/archive/06/pcd0600r.png'
    # filename = '/home/aby/Workspace/Cornell/archive/07/pcd0754z.png'
    # # filename = '/home/aby/Workspace/parallel-jaw-grasping-dataset/data/heightmap-color/000000.png'
    # # from generators.cornell import load_and_preprocess_img
    # # test_data = load_and_preprocess_img(filename, side_after_crop=None, resize_height=512, resize_width=512)
    # from dataset_processing import image
    # from dataset_processing.cornell_generator import CornellDataset
    # from dataset_processing.grasp import Grasp
    # test_data = CornellDataset.load_custom_image(filename, zoom_fac=1.0)
    # test_data = np.array(test_data)
    # test_data = test_data[np.newaxis, ...]
    # print(' ### TEST ###: ', test_data.shape)
    # model_out = model.predict(test_data, verbose=1, steps=1)
    # test_out = model_out[0]
    # Y_out = np.copy(model_out[0])
    # # print(len(test_out))
    # # print(type(test_out[0]))
    # # print(model.layers['grasp_5'].output)
    # print(test_out.shape)
    # # print("Output vector: ", test_out)
    # output_vectors = []
    # for i in range(test_out.shape[0]):
    #     pred_grasp = Grasp(test_out[i][0:2], *test_out[i][2:], unnorm=True)
    #     output_vectors.append(pred_grasp.as_bbox)
    # print(output_vectors)
    # print('MAX YAHA AYA', max(output_vectors))
    # # Remove batch dim

    # all_predictions = Y_out[:,0:6]
    # all_score = Y_out[:,6]
    
    # DISPLAY_PRED = 50
    # # Sort y_out based on score
    # sort_index = (-all_score).argsort()
    # all_score = all_score[sort_index]
    # all_predictions = all_predictions[sort_index,:]
    # print('SCORE: ', all_score[:DISPLAY_PRED])
    # import matplotlib
    # import matplotlib.pyplot as plt
    # ### Plot all grasps
    # fig, ax = plt.subplots(1,2)
    # fig.set_size_inches(10, 5)
    # ## Show Original Image
    # ax[0].imshow(test_data[0,:,:,:])
    # ax[0].set_title('Original Image')
    # ## Show RGD image and set axis title
    # ax[1].imshow(test_data[0,:,:,:])
    # ax[1].set_title('Predicted Grasps')
    # ## Show Predicted grasps on orig image
    # colormap=plt.get_cmap('cool')
    # display_score = all_score[:DISPLAY_PRED]
    # center_y = []
    # center_x = []
    # # col_norm = matplotlib.colors.Normalize(vmin=display_score[-1], vmax=display_score[0], clip=False)
    # col_norm = matplotlib.colors.Normalize(vmin=min(display_score), vmax=max(display_score), clip=False)
    # for j in range(min(all_predictions.shape[0], DISPLAY_PRED)):
    #     plot_grasp = Grasp(all_predictions[j][0:2], *all_predictions[j][2:6], quality=display_score[j], unnorm=True)
    #     plot_points = plot_grasp.as_gr.points
    #     print('PLOT POINTS: ', plot_points)
    #     points = np.vstack((plot_points, plot_points[0]))
    #     ax[1].plot(points[:, 1], points[:, 0], color=colormap(col_norm(display_score[j])))
    #     center_y.append(plot_grasp.center[0])
    #     center_x.append(plot_grasp.center[1])
    # sc=ax[1].scatter(center_x, center_y, c=display_score, cmap=colormap)
    # # ax[1][1].scatter(center_x, center_y, c=display_score, cmap=colormap)
    # plt.colorbar(sc)
    # plt.show()
    # exit()
    # ## TEST ON SINGLE IMAGE END


    print("\nStarting Training!\n")
    history = model.fit_generator(
        generator = train_generator,
        # steps_per_epoch = 2,
        steps_per_epoch = len(train_generator),
        initial_epoch = args.start_epoch,
        epochs = args.epochs,
        # epochs = 1,
        verbose = 1,
        callbacks = callbacks,
        workers = args.workers,
        use_multiprocessing = args.multiprocessing,
        max_queue_size = args.max_queue_size,
        validation_data = validation_generator
    )
    print("\nTraining Complete! Saving...\n")
    os.makedirs(args.snapshot_path, exist_ok = True)
    # NOT WORKING
    model.save_weights(os.path.join(args.snapshot_path, '{dataset_type}_finish.h5'.format(dataset_type = args.dataset_type)))

    # Calculating Training time
    end_time = datetime.now()
    print('Training Duration: {}'.format(end_time - start_time))
    print("\nEnd of Code...\n")
    


def create_generators(args):
    """
    Create generators for training and validation.
    Args:
        args: parseargs object containing configuration for generators.
    Returns:
        The training and validation generators.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }
    
    if args.dataset_type == 'cornell':
        dataset = args.cornell_path
        # open output file for reading
        # with open(dataset+'/train_overfit_2.txt', 'r') as filehandle:
        with open(dataset+'/train_1.txt', 'r') as filehandle:
            train_data = json.load(filehandle)
        # with open(dataset+'/valid_overfit_2.txt', 'r') as filehandle:
        with open(dataset+'/valid_1.txt', 'r') as filehandle:
            valid_data = json.load(filehandle)
        
        train_generator = CornellDataset(
            dataset,
            train_data,
            # train=False, ## COMMENT IT FOR MAIN TRAINING
            **common_args
        )

        validation_generator = CornellDataset(
            dataset,
            valid_data,
            train=False,
            batch_size=1,
            phi=args.phi,
        )
    elif args.dataset_type == 'amazon':
        dataset = args.amazon_path
        with open(dataset+'/train-split.txt', 'r') as filehandle:
            lines = filehandle.readlines()
            train_data = []
            for line in lines:
                train_data.append(line.strip())
        with open(dataset+'/test-split.txt', 'r') as filehandle:
            lines = filehandle.readlines()
            valid_data = []
            for line in lines:
                valid_data.append(line.strip())
    elif args.dataset_type == 'vmrd':
        dataset = args.vmrd_path
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

        train_generator = VMRDDataset(
            dataset,
            train_data,
            **common_args
        )

        validation_generator = VMRDDataset(
            dataset,
            valid_data,
            train=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator

def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args:
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None
    
    if args.dataset_type == "cornell" or args.dataset_type == "amazon" or args.dataset_type == "vmrd":
        snapshot_path = args.snapshot_path
        # save_path = args.validation_image_save_path
        tensorboard_dir = args.tensorboard_dir
        metric_to_monitor = "val_loss"
        mode = "min"
        # metric_to_monitor = "grasp_accuracy"
        # mode = "max"
    else:
        snapshot_path = args.snapshot_path
        # save_path = args.validation_image_save_path
        tensorboard_dir = args.tensorboard_dir
        
    # if save_path:
    #     os.makedirs(save_path, exist_ok = True)

    if tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir = tensorboard_dir,
            histogram_freq = 1,
            write_graph = False,
            write_grads = False,
            write_images = False,
            embeddings_freq = 0,
            embeddings_layer_names = None,
            embeddings_metadata = None,
            profile_batch = '25,30'
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        from eval.eval_callback import Evaluate
        evaluation = Evaluate(validation_generator, prediction_model, tensorboard = tensorboard_callback, multi_pred=True)
        evaluation._supports_tf_logs = True
        callbacks.append(evaluation)

    # Learning Rate Schedule
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.5,
        patience   = 5,
        verbose    = 1,
        mode       = 'min',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 1e-7
    ))
    
    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(snapshot_path, exist_ok = True)
        # checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(snapshot_path, 'phi_{phi}_{dataset_type}_best_{metric}.h5'.format(phi = str(args.phi), metric = metric_to_monitor, dataset_type = args.dataset_type)),
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(snapshot_path, '{dataset_type}_best_{metric}.h5'.format(phi = str(args.phi), metric = metric_to_monitor, dataset_type = args.dataset_type)),
                                                     verbose = 1,
                                                     save_weights_only = True,
                                                     save_best_only = True,
                                                     monitor = metric_to_monitor,
                                                     save_freq='epoch',
                                                     mode = mode)
        # checkpoint._supports_tf_logs = False
        callbacks.append(checkpoint)

    return callbacks

if __name__ == '__main__':
    main()