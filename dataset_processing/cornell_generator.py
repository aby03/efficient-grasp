import glob
import os

from dataset_processing import image
from dataset_processing import grasp as gp
# import image        # For Debugging
# import grasp as gp  # For Debugging

import numpy as np
import random
from tensorflow.python.keras.utils.data_utils import Sequence
from utils.anchors import anchors_for_shape, anchor_targets_bbox, AnchorParameters

#Debug
import json

class CornellDataset(Sequence):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, dataset_path, list_IDs, phi=0, batch_size=1, output_size=512, n_channels=3,
                 n_classes=12, shuffle=True, train=True, run_test=False):
        """
        :param dataset_path: Cornell Dataset directory.
        :param list_IDs: List of the image files for the generator.
        :param phi: (NOT USED CURRENTLY)
        :param output_size: Image output size in pixels (square)
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param shuffle: Shuffle list of files after every epoch
        :param train: Whether its train generator or valid generator
        :param run_test: (TO WRITE)
        """
        self.run_test = run_test
        self.random_rotate = True   # Used only when train is enabled for data aug
        self.random_zoom = True     # Used only when train is enabled for data aug

        # Generator
        self.output_size = output_size
        if train:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        self.dataset = dataset_path
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train = train
        self.init_shape = None

        # List of rgd files of train/valid split
        self.rgd_files = list_IDs
        # List of grasp files
        self.grasp_files = [f.replace('z.png', 'cpos.txt') for f in self.rgd_files]

        # FOR RGB IMAGE
        # self.rgd_files = [f.replace('z.png', 'r.png') for f in self.rgd_files]
        self.length = len(self.grasp_files)
        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(dataset_path))

        self.anchor_parameters = AnchorParameters.default
        self.anchors = anchors_for_shape((self.output_size, self.output_size), anchor_params = self.anchor_parameters)
        self.indexes = np.arange(len(self.rgd_files))
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.rgd_files) / self.batch_size))

    def size(self):
        """ Size of the dataset.
        """
        return len(self.rgd_files)
    
    def num_classes(self):
        """ Number of angle classes (12 for 15 degree angles)
        """
        return self.n_classes

    def _get_crop_attrs(self, idx):
        gtbbs = gp.GraspRectangles.load_from_cornell_file(self.dataset+self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, self.init_shape[0] - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, self.init_shape[1] - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = gp.GraspRectangles.load_from_cornell_file(self.dataset+self.grasp_files[idx])

        ## Offset to crop to min(height, width)
        left = max(0, self.init_shape[1] - self.init_shape[0])//2
        top = max(0, self.init_shape[0] - self.init_shape[1])//2
        gtbbs.offset((-top, -left))

        ## Perform zoom about cropped image center
        side = min(self.init_shape[0], self.init_shape[1])
        gtbbs.zoom(zoom, (side // 2, side // 2))
        
        ## Side scale points to simulate resizing of image
        gtbbs.corner_scale( (self.output_size/side, self.output_size/side) )       
        
        ## Rotate along final center
        gtbbs.rotate(rot, (self.output_size//2, self.output_size//2))

        return gtbbs

    def get_gen_inp_img(self, idx, rot=0, zoom=1.0, normalise=True):
        rgd_img = image.Image.from_file(self.dataset+self.rgd_files[idx])
        self.init_shape = rgd_img.img.shape

        ## Perform crop to min(height,width)
        left = max(0, self.init_shape[1] - self.init_shape[0])//2
        top = max(0, self.init_shape[0] - self.init_shape[1])//2
        top_left = (top, left)
        bottom_right = (self.init_shape[0]-top, self.init_shape[1]-left)
        rgd_img.crop(top_left, bottom_right)

        ## Perform central zoom
        rgd_img.zoom(zoom)

        ## Resizing (Side scaling)
        rgd_img.resize((self.output_size, self.output_size))

        ## Rotate about final center
        rgd_img.rotate(rot, (self.output_size//2, self.output_size//2))

        if normalise:
            rgd_img.normalise()
            # rgd_img.img = rgd_img.img.transpose((2, 0, 1))
        return rgd_img.img#, scale

    # def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
    #     rgd_img = image.Image.from_file(self.dataset+self.rgd_files[idx].replace('z.png','r.png'))
    #     center, left, top = self._get_crop_attrs(idx)
    #     rgd_img.rotate(rot, center)
    #     rgd_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
    #     rgd_img.zoom(zoom)
    #     rgd_img.resize((self.output_size, self.output_size))
    #     if normalise:
    #         rgd_img.normalise()
    #         # rgd_img.img = rgd_img.img.transpose((2, 0, 1))
    #     return rgd_img.img

    @staticmethod
    def load_custom_image(filename, zoom_fac=1.0, output_size=512, normalise=True):
        rgd_img = image.Image.from_file(filename)
        rgd_img.resize((output_size, output_size))
        if zoom_fac != 1.0:
            rgd_img.zoom(zoom_fac)
        if normalise:
            rgd_img.normalise()
            # rgd_img.img = rgd_img.img.transpose((2, 0, 1))
        return rgd_img.img

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y_g = self.__data_generation(indexes)

        return X, y_g    

    def get_annotation_val(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, annotation_groups = self.__get_valid_item(indexes)
        return X, annotation_groups

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X, annotations_groups = self.__get_train_item(indexes)
        return X, annotations_groups

        # if self.train:
        #     X, annotations_groups = self.__get_train_item(indexes)
        #     return X, annotations_groups
        # else:
        #     # If validation data
        #     X, annotations_groups = self.__get_valid_item(indexes)

        #     annotations_groups = np.asarray(annotations_groups)
        #     return X, annotations_groups

    def __get_train_item(self, indexes):
        X = np.empty((self.batch_size, self.output_size, self.output_size, self.n_channels))
        annotations_group = []
        # For every image in batch
        for i in range(indexes.shape[0]):
            # Rotation augmentation
            if self.train and self.random_rotate:
                rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
                rot = random.choice(rotations)
            else:
                rot = 0.0
            # Zoom Augmentation
            if self.train and self.random_zoom:
                zoom_factor = np.random.uniform(0.5, 0.875)
            else:
                zoom_factor = 0.875
            # Load image with zoom and rotation
            rgd_img = self.get_gen_inp_img(indexes[i], rot, zoom_factor)
            # Load bboxes
            gtbb = self.get_gtbb(indexes[i], rot, zoom_factor)
            # Pick all grasps
            annotations = { 'labels': np.zeros((len(gtbb.grs),)),
                            'bboxes': np.zeros((len(gtbb.grs), 4))}
            for g_id in range(len(gtbb.grs)):
                # Get Grasp as [xmin, ymin, xmax, ymax] and angle
                grasp_vec = gtbb[g_id].as_grasp
                grasp_bbox = grasp_vec.as_horizontal_bbox
                grasp_angle = grasp_vec.as_angle
                # print('BBOX: ', grasp_bbox)
                # print('ANGLE: ', grasp_angle)
                # print('ANGLE CLASS: ', int(round( (grasp_angle+np.pi/2) / (np.pi / 12))))
                # print('ANGLE CLASS: ', int(round( (grasp_angle+np.pi/2) / (np.pi / 12))) % 12)
                '''Generate annotations
                    labels: theta label (0, 15, 30, ..., 165) -> len = 12
                    bboxes: x1, y1, x2, y2 <- get from x y h w
                '''
                # Angle Class: 0-11 for every 15 degree angle
                annotations['labels'][g_id] = int(round( (grasp_angle+np.pi/2) / (np.pi / 12))) % 12
                annotations['bboxes'][g_id,:] = grasp_bbox
            # Store Image
            X[i,] = rgd_img
            # Store each annotation for an image
            annotations_group.append(annotations)
        
        # Generate Target anchors
        batches_targets = anchor_targets_bbox(
            self.anchors,
            X,
            annotations_group,
            self.n_classes        # num of classes = 12 (based on angle theta every 15 degree is a class) (1 is bg class)
        )
        # print ('Label Batches: ', batches_targets[0].shape) ## Label Batches:  (1, 49104, 13)
        # print ('BBox targets batches: ', batches_targets[1].shape) ## BBox targets batches:  (1, 49104, 5)

        ## Debug Start
        # # Display all Grasps
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        # ax.imshow(rgd_img)
        # gtbb.plot(ax, 1)
        # plt.show()
        ## Debug end  
        return X, batches_targets
    
    def __get_valid_item(self, indexes):
        X = np.empty((self.batch_size, self.output_size, self.output_size, self.n_channels))
        annotations_group = []
        # For every image in batch
        for i in range(indexes.shape[0]):
            # Rotation augmentation
            rot = 0.0
            # Zoom Augmentation
            zoom_factor = 0.875
            # Load image with zoom and rotation
            rgd_img = self.get_gen_inp_img(indexes[i], rot, zoom_factor)
            # Load bboxes
            gtbb = self.get_gtbb(indexes[i], rot, zoom_factor)
            # Pick all grasps
            annotations = { 'labels': np.zeros((len(gtbb.grs),)),
                            'bboxes': np.zeros((len(gtbb.grs), 4))}
            for g_id in range(len(gtbb.grs)):
                # Get Grasp as [xmin, ymin, xmax, ymax] and angle
                grasp_vec = gtbb[g_id].as_grasp
                grasp_bbox = grasp_vec.as_horizontal_bbox
                grasp_angle = grasp_vec.as_angle
                # print('BBOX: ', grasp_bbox)
                # print('ANGLE: ', grasp_angle)
                # print('ANGLE CLASS: ', int(round( (grasp_angle+np.pi/2) / (np.pi / 12))))
                # print('ANGLE CLASS: ', int(round( (grasp_angle+np.pi/2) / (np.pi / 12))) % 12)
                '''Generate annotations
                    labels: theta label (0, 15, 30, ..., 165) -> len = 12
                    bboxes: x1, y1, x2, y2 <- get from x y h w
                '''
                # Angle Class: 0-11 for every 15 degree angle
                annotations['labels'][g_id] = int(round( (grasp_angle+np.pi/2) / (np.pi / 12))) % 12
                annotations['bboxes'][g_id,:] = grasp_bbox
            # Store Image
            X[i,] = rgd_img
            # Store each annotation for an image
            annotations_group.append(annotations)

        # X = np.empty((self.batch_size, self.output_size, self.output_size, self.n_channels))
        # annotations_group = []
        # # For every image in batch
        # for i in range(indexes.shape[0]):
        #     # Load image with 1 zoom and 0 rotation
        #     rgd_img = self.get_gen_inp_img(indexes[i], 0, 0.875)
        #     # Load bboxes
        #     gtbb = self.get_gtbb(indexes[i], 0, 0.875)
        #     # Pick all grasps
        #     y_grasp_image = []
        #     # Pad count
        #     count = 30
        #     for g_id in range(len(gtbb.grs)):
        #         # Get Grasp as list [y x sin_t cos_t h w] AFTER NORMALIZATION
        #         grasp = (gtbb[g_id].as_grasp).as_list
        #         # Store each grasp for an image
        #         y_grasp_image.append(grasp)
        #         count -= 1
        #         if count == 0:
        #             break
        #     while (count > 0):
        #         pad_0 = [1e8, 1e8, 1e8, 1e8, 1e8, 1e8]
        #         y_grasp_image.append(pad_0)
        #         count -= 1
        #     # Store Image
        #     X[i,] = rgd_img
        #     # Store each annotation for an image
        #     annotations_group.append(y_grasp_image)
                        
        ## Debug Start
        # # Display all Grasps
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        # ax.imshow(rgd_img)
        # gtbb.plot(ax, 1)
        # plt.show()
        ## Debug end  
        return X, annotations_group

# ### TESTING
# dataset = "/home/aby/Workspace/Cornell/archive"
# with open(dataset+'/train_1.txt', 'r') as filehandle:
#     train_data = json.load(filehandle)

# train_generator = CornellDataset(
#     dataset,
#     train_data,
#     train=True,
#     shuffle=False,
#     batch_size=1
# )

# for i in range(0, 20):
#     train_generator[i]