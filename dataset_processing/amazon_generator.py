import glob
import os

from dataset_processing import image
from dataset_processing import grasp as gp
# import image        # For Debugging
# import grasp as gp  # For Debugging

import numpy as np
import random
from tensorflow.python.keras.utils.data_utils import Sequence

#Debug
import json

class AmazonDataset(Sequence):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, dataset_path, list_IDs, phi=0, batch_size=1, output_size=512, n_channels=3,
                 n_classes=10, shuffle=True, train=True, run_test=False):
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
        self.random_zoom = False     # Used only when train is enabled for data aug

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
        self.rgd_files = ['/heightmap-color/' + f + '.png' for f in list_IDs]
        # List of grasp files
        self.grasp_files = ['/label/' + f + '.good.txt' for f in list_IDs]
        
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(dataset_path))

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

    def _get_crop_attrs(self, idx):
        gtbbs = gp.GraspRectangles.load_from_cornell_file(self.dataset+self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = gp.GraspRectangles.load_from_amazon_file(self.dataset+self.grasp_files[idx])
        
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

        # print('T1: ', gtbbs[0])
        # center, left, top = self._get_crop_attrs(idx)
        # center = [rgd_img.img.shape[0]//2, rgd_img.img.shape[1]//2]
        # gtbbs.offset((-top, -left))
        # if not zoom == 1.0:
        #     gtbbs.zoom(zoom, center)
        # gtbbs.corner_scale(scale)
        # # Rotate from new center
        # gtbbs.rotate(rot, (self.output_size //2, self.output_size //2))
        
        return gtbbs

    def get_rgd(self, idx, rot=0, zoom=1.0, normalise=True):
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


        # print('SHAPE: ', rgd_img.img.shape)
        # center = [rgd_img.img.shape[0]//2, rgd_img.img.shape[1]//2]
        # print('SHAPE: ', center)
        # center, left, top = self._get_crop_attrs(idx)
        # rgd_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # if not zoom == 1.0:
        #     rgd_img.zoom(zoom)
        # # Scale (y,x)
        # scale = (self.output_size/rgd_img.img.shape[0],self.output_size/rgd_img.img.shape[1])
        # rgd_img.resize((self.output_size, self.output_size))
        # rgd_img.rotate(rot, center)
        if normalise:
            rgd_img.normalise()
            # rgd_img.img = rgd_img.img.transpose((2, 0, 1))
        return rgd_img.img#, scale

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgd_img = image.Image.from_file(self.dataset+self.rgd_files[idx].replace('z.png','r.png'))
        center, left, top = self._get_crop_attrs(idx)
        rgd_img.rotate(rot, center)
        rgd_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgd_img.zoom(zoom)
        rgd_img.resize((self.output_size, self.output_size))
        if normalise:
            rgd_img.normalise()
            # rgd_img.img = rgd_img.img.transpose((2, 0, 1))
        return rgd_img.img

    @staticmethod
    def load_custom_image(filename, output_size=512, normalise=True):
        rgd_img = image.Image.from_file(filename)
        rgd_img.resize((output_size, output_size))
        rgd_img.zoom(0.75)
        if normalise:
            rgd_img.normalise()
            # rgd_img.img = rgd_img.img.transpose((2, 0, 1))
        return rgd_img.img

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        # print(self.rgd_files[indexes[0]])
        X, y_g = self.__data_generation(indexes)

        return X, y_g    

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.output_size, self.output_size, self.n_channels))
        y_grasp = []

        # For every image in batch
        for i in range(indexes.shape[0]):
            if self.train:
                # If training data
                # Rotation augmentation
                if self.random_rotate:
                    rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
                    rot = random.choice(rotations)
                else:
                    rot = 0.0
                # Zoom Augmentation
                if self.random_zoom:
                    zoom_factor = np.random.uniform(0.5, 0.875)
                else:
                    zoom_factor = 1.0
                
                # Load image with zoom and rotation
                rgd_img = self.get_rgd(indexes[i], rot, zoom_factor)

                # Load bboxes
                gtbb = self.get_gtbb(indexes[i], rot, zoom_factor)
                # Pick all grasps
                y_grasp_image = []
                # Pad count
                count = 100
                for g_id in range(len(gtbb.grs)):
                    # Get Grasp as list [y x sin_t cos_t h w] AFTER NORMALIZATION
                    grasp = (gtbb[g_id].as_grasp).as_list
                    # Store each grasp for an image
                    y_grasp_image.append(grasp)
                    count -= 1
                    if count == 0:
                        break
                while (count > 0):
                    # pad_0 = [1e8, 1e8, 1e8, 1e8, 1e8, 1e8]
                    pad_0 = [1e16, 1e16, 1e16, 1e16, 1e16, 1e16]
                    # pad_0 = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]
                    y_grasp_image.append(pad_0)
                    count -= 1
                # Store all grasps for an image
                y_grasp.append(y_grasp_image)

            else:
                # If validation data
                # Load image with 1 zoom and 0 rotation
                rgd_img = self.get_rgd(indexes[i], 0, 1.0)

                # Load bboxes
                gtbb = self.get_gtbb(indexes[i], 0, 1.0)
                # Pick all grasps
                y_grasp_image = []
                # Pad count
                count = 100
                for g_id in range(len(gtbb.grs)):
                    # Get Grasp as list [y x sin_t cos_t h w] AFTER NORMALIZATION
                    grasp = (gtbb[g_id].as_grasp).as_list
                    # Store each grasp for an image
                    y_grasp_image.append(grasp)
                    count -= 1
                    if count == 0:
                        break
                while (count > 0):
                    pad_0 = [1e16, 1e16, 1e16, 1e16, 1e16, 1e16]
                    y_grasp_image.append(pad_0)
                    count -= 1
                # Store all grasps for an image
                y_grasp.append(y_grasp_image)
                ## OLD Val START
                # for g_id in range(len(gtbb.grs)):
                #     # Get Grasp as list [y x sin_t cos_t h w] AFTER NORMALIZATION
                #     grasp = (gtbb[g_id].as_grasp).as_list 
                #     # Store each grasp for an image
                #     y_grasp_image.append(grasp)

                # # Store all grasps for an image
                # y_grasp.append(y_grasp_image)
                ## OLD Val END
            ## Debug start
            # # Display all Grasps
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_axes([0,0,1,1])
            # ax.imshow(rgd_img)
            # gtbb.plot(ax, 1)
            # plt.show()
            ## Debug end
            # Store Image sample
            X[i,] = rgd_img
        if not self.run_test:
            yy = np.asarray(y_grasp)
            # print("debug: ", yy.shape)
            return X, yy
        else:
            # Return RGB image for displaying
            X_rgb = np.empty((self.batch_size, self.output_size, self.output_size, self.n_channels))
            for i in range(indexes.shape[0]):
                X_rgb[i,] = self.get_rgb(indexes[i], 0, 0.875, normalise=False)

            return [X, X_rgb], gtbb

# ### TESTING
# dataset = "/home/aby/Workspace/parallel-jaw-grasping-dataset/data"
# with open(dataset+'/test-split.txt', 'r') as filehandle:
#     lines = filehandle.readlines()
#     train_data = []
#     for line in lines:
#         train_data.append(line.strip())
#     # train_data = json.load(filehandle)

# train_generator = AmazonDataset(
#     dataset,
#     train_data,
#     train=False,
#     shuffle=False,
#     batch_size=1
# )

# for i in range(0, 20):
#     x, y = train_generator[i]
    # print(y)
    # break