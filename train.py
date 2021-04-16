import argparse
from dataset_processing.cornell_generator import CornellDataset

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
    parser.add_argument('--workers', help = 'Number of generator workers.', type = int, default = 4)
    parser.add_argument('--max-queue-size', help = 'Queue length for multiprocessing workers in fit_generator.', type = int, default = 10)
    
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)

def main(args = None):
    """
    Train an EfficientGrasp model.

    Args:
        args: parseargs object containing configuration for the training procedure.
    """

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generators
    print("\nCreating the Generators...")
    train_generator, validation_generator = create_generators(args)
    print("Done!")

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
        with open(dataset+'/train_1.txt', 'r') as filehandle:
            train_data = json.load(filehandle)
        with open(dataset+'/valid_1.txt', 'r') as filehandle:
            valid_data = json.load(filehandle)
        
        train_generator = CornellDataset(
            dataset,
            train_data,
            **common_args
        )

        validation_generator = CornellDataset(
            dataset,
            valid_data,
            train=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator