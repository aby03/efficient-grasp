# efficient-grasp
EfficientGrasp model based on EfficientPose and EfficientDet to identify robotic grasps.
## Installation

1. Install Ubuntu 20.04

2. Install cuda using deb local install (CUDA 11.4) from the following link.

> https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile

3. Install libcudnn8

4. Install miniconda. Create new conda environment.
> conda create --name efficient-grasp
> conda activate efficient-grasp
> conda install tensorflow-gpu
> conda install -c conda-forge opencv
> conda install -c conda-forge matplotlib
> conda install -c conda-forge imageio
> conda install -c anaconda scikit-image 
> conda install -c conda-forge keras-applications
> conda install -c conda-forge tqdm 
> conda install -c conda-forge shapely

4. (Alternate) Unpack conda environment following this: https://www.anaconda.com/blog/moving-conda-environments

### Anchors
1. To compile compute_overlap 
> python setup.py build_ext --inplace

## Run Training

1. VSCode

    a. Go to train.py

    b. Press F5 or Ctrl-F5

2. Terminal

    > python train.py --phi 0 --batch-size 1 --lr 1e-4 --weights imagenet cornell /home/aby/Workspace/MTP/Datasets/Cornell/archive

## Run Prediction
1. Set model and dataset or custom images to run inference on in inference_dataset.py

2a. VSCode
    
    a. Go to inference_dataset.py

    b. Press F5 or Ctrl-F5

2b. Terminal

    > python inference_dataset.py

<!-- ## Ros commands

### Create Package
1. > cd ros-grasp/src
2. > catkin_create_pkg beginner_tutorials std_msgs rospy roscpp
3. > cd ..
4. > catkin_make

### Add python files to package
1. Go to ros-grasp/src/<PACKAGE-NAME>
2. mkdir scripts
3. cd scripts
4. ADD ALL PYTHON FILES HERE
5. cd ..
6. ADD NAMES OF ALL PYTHON FILES IN CMAKELIST HERE IN "catkin_install_python()"

### Building package
1. cd ros-grasp     (CATKIN WS FOLDER)
2. catkin_make
3. source devel/setup.bash -->