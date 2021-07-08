# efficient-grasp
EfficientGrasp model based on EfficientPose and EfficientDet to identify robotic grasps. ROS simulation included.

## Pre-requisuite
1. Ubuntu 20.04

2. CUDA (11.4 installed using deb installer)
> https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile

3. miniconda

## Installation (with ROS)
1. Create conda environment and install ros(source: https://github.com/RoboStack/ros-noetic)
> conda create -n effgrasp-ros python=3.8  
> conda activate effgrasp-ros  
> conda config --env --add channels conda-forge  
> conda config --env --add channels robostack  
> conda config --env --set channel_priority strict  
> conda install ros-noetic-desktop  
> conda install compilers cmake pkg-config make ninja  
> conda install catkin_tools
> rosdep init  s
> rosdep update  

2. Create catkin workspace for ros packages (named: effgrasp_ros).
> mkdir -p ~/Workspace/MTP/effgrasp_ros/src  
> cd ~/Workspace/MTP/effgrasp_ros
> catkin_make  

3. Source ros environment over current environment. (Do this step after building each new ros package)
> cd ~/Workspace/MTP/effgrasp_ros  
> source devel/setup.bash

4. Clone efficient-grasp
> cd ~/Workspace/MTP  
> git clone https://github.com/aby03/efficient-grasp  

5. Install efficient-grasp dependencies
> conda install tensorflow-gpu  
> conda install cudatoolkit=10.1.243 cudnn=7.6.5
> conda install -c conda-forge opencv matplotlib imageio keras-applications tqdm shapely  
> conda install scikit-image 
> conda install cython
## Installation
1. Clone and move into repository directory.
> git clone https://github.com/aby03/efficient-grasp  
> cd efficient-grasp

2. Create conda environment and "effgrasp-ros" and install dependencies.
> conda env create --file environment.yml  
> conda activate efficient-grasp


##### TO DO: conda env update --file environment.yml
##### ROS path: /home/aby03/miniconda3/envs/effgrasp-ros/etc/ros/
2. (Alternate: Do this in place of above if you want to install each dependency manually)  
Create new conda environment and install the dependencies.
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

3. Run the following command.
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