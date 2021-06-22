from functools import reduce

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import backend
# from tensorflow.keras import regularizers
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization, RegressTranslation, CalculateTxTy, GroupNormalization
from initializers import PriorProbability
from utils.anchors import anchors_for_shape
import numpy as np


MOMENTUM = 0.997
EPSILON = 1e-4

def get_scaled_parameters_multi(phi):
    """
    Get all needed scaled parameters to build EfficientGrasp
    Args:
        phi: EfficientGrasp scaling hyperparameter phi
    
    Returns:
       Dictionary containing the scaled parameters
    """
    #info tuples with scalable parameters
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)

    bifpn_widths = (144, 60, 96)   # 144
    bifpn_depths = (3, 3, 3)                # 3
    subnet_depths = (3, 3, 4)               # 4
    subnet_width = (96, 36, 48)    # 96
    subnet_iteration_steps = (2, 1, 2)      # 2
    num_groups_gn = (6, 3, 3)           # 6  #try to get 16 channels per group ## width > groups * 16 

    # bifpn_widths = (64, 88, 112, 160, 224, 288, 384)
    # bifpn_depths = (3, 4, 5, 6, 7, 7, 8)
    # subnet_depths = (3, 3, 3, 4, 4, 4, 5)
    # subnet_width = (96, 88, 112, 160, 224, 288, 384)
    # subnet_iteration_steps = (2, 1, 1, 2, 2, 2, 3)
    # num_groups_gn = (4, 4, 7, 10, 14, 18, 24) #try to get 16 channels per group
    backbones = (EfficientNetB0,
                 EfficientNetB1,
                 EfficientNetB2,
                 EfficientNetB3,
                 EfficientNetB4,
                 EfficientNetB5,
                 EfficientNetB6)
    
    parameters = {"input_size": image_sizes[phi],
                  "bifpn_width": bifpn_widths[phi],
                  "bifpn_depth": bifpn_depths[phi],
                  "subnet_depth": subnet_depths[phi],
                  "subnet_width": subnet_width[phi],
                  "subnet_num_iteration_steps": subnet_iteration_steps[phi],
                  "num_groups_gn": num_groups_gn[phi],
                  "backbone_class": backbones[phi]}    
    return parameters

def build_EfficientGrasp_multi(phi,
                        num_anchors = 9,
                        num_classes = 12,
                        score_threshold = 0.5,
                        anchor_parameters = None,
                        freeze_bn = False,
                        print_architecture = False):
    # Get Parameters for model
    assert phi in range(7)
    scaled_parameters = get_scaled_parameters_multi(phi)
    
    input_size = scaled_parameters["input_size"]
    input_shape = (input_size, input_size, 3)
    backbone_class = scaled_parameters["backbone_class"]
    bifpn_width = scaled_parameters["bifpn_width"]
    subnet_width = scaled_parameters["subnet_width"]
    bifpn_depth = scaled_parameters["bifpn_depth"]
    subnet_depth = scaled_parameters["subnet_depth"]
    subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
    num_groups_gn = scaled_parameters["num_groups_gn"]

    output_dim = 6
    pred_count = 100

    # Input layer
    image_input = layers.Input(input_shape)
    
    # 1. Build EfficientNet backbone
    backbone_feature_maps = backbone_class(input_tensor = image_input)

    # 2. Build BiFPN
    fpn_feature_maps = build_BiFPN(backbone_feature_maps, bifpn_depth, bifpn_width, num_groups_gn, freeze_bn)

    # 3. Build GraspNet
    angle_net = AngleNet(subnet_width,
                        subnet_depth,
                        num_classes = num_classes,
                        num_anchors = num_anchors,
                        freeze_bn = freeze_bn,
                        name = 'angle_net')
    grasp_net = GraspNet(subnet_width,
                        subnet_depth,
                        subnet_num_iteration_steps,
                        use_group_norm = True,
                        num_groups_gn = num_groups_gn,
                        num_anchors = num_anchors,
                        freeze_bn=freeze_bn, 
                        name='grasp_net')
    
    # # Get reshape dims for applying grasp net
    # reshape_dim = 0     # 64x64 + 32x32 + 16x16 + 8x8 + 4x4
    # for i in range(len(fpn_feature_maps)):
    #     reshape_dim += fpn_feature_maps[i].shape[1]*fpn_feature_maps[i].shape[2]
    
    # Apply Angle Net to get angle class
    angle_classification = [angle_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    angle_classification = layers.Concatenate(axis=1, name='angle_classification')(angle_classification)

    # Apply GraspNet to get grasp bounding box
    grasp_regression = [grasp_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    grasp_bbox_regression = layers.Concatenate(axis=1, name='bbox_regression')(grasp_regression)

    #get anchors and apply predicted translation offsets to translation anchors
    anchors = anchors_for_shape((input_size, input_size), anchor_params = anchor_parameters)
    
    # apply predicted 2D bbox regression to anchors
    anchors_input = np.expand_dims(anchors, axis = 0)
    bboxes = RegressBoxes(name='boxes')([anchors_input, grasp_bbox_regression[..., :4]])
    bboxes = ClipBoxes(name='clipped_boxes')([image_input, bboxes])
    
    ## Train Output Shape -> [(None, 49104, 13), (None, 49104, 6)]
    efficientgrasp_train = models.Model(inputs = [image_input], outputs = [angle_classification, grasp_bbox_regression], name = 'efficientgrasp_train')
    
    # filter detections (apply NMS / score threshold / select top-k)
    filtered_detections = FilterDetections(     name = 'filtered_detections',
                                                class_specific_filter = False,
                                                score_threshold = score_threshold
                                           )([bboxes, angle_classification])
    
    ## Prediction Output shape -> [(None, 100, 4), (None, 100), (None, 100)] -> (bbox, score, angle_class)
    efficientgrasp_prediction = models.Model(inputs = [image_input], outputs = filtered_detections, name = 'efficientpose_prediction')
    ''' OLD MODEL
    grasp_regression = layers.Reshape((-1,reshape_dim * output_dim))(grasp_regression) # 5456 for num_anchors=1 && 49104 for 9 

    grasp_regression_multi = layers.Dense(pred_count * output_dim, name='regression_grasp'
                                        # , kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=256.0)
                                        , kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=50.0)
                                        )(grasp_regression)
    grasp_regression_multi = layers.Reshape((pred_count, output_dim), name='regression_grasp_re')(grasp_regression_multi)
    grasp_regression_score = layers.Dense(pred_count, name='regression_score', activation=tf.keras.activations.sigmoid)(grasp_regression)
    grasp_regression_score = layers.Reshape((pred_count,1), name='regression_score_re')(grasp_regression_score)

    grasp_regression_out = layers.Concatenate(axis=2, name='regression_out')([grasp_regression_multi, grasp_regression_score])

    # Build Complete Model
    efficientgrasp_train = models.Model(inputs = [image_input], outputs = grasp_regression_out, name = 'efficientgrasp')
    efficientgrasp_prediction = models.Model(inputs = [image_input], outputs = grasp_regression_out, name = 'efficientgrasp_prediction')
    '''
    
    all_layers = list(set(efficientgrasp_train.layers + efficientgrasp_prediction.layers))

    if print_architecture:
        # freeze backbone layers
        # 227, 329, 329, 374, 464, 566, 656 -> based on EfficientNet chosen. 227 for B0
        
        ## Train Arch
        # for i in range(1, 227):
        #     efficientgrasp_train.layers[i].trainable = False
        # efficientgrasp_train.summary()
        # print(efficientgrasp_train.layers[-2].output_shape)
        # print(efficientgrasp_train.layers[-1].output_shape)
        
        ## Prediction Arch
        for i in range(1, 227):
            efficientgrasp_prediction.layers[i].trainable = False
        efficientgrasp_prediction.summary()
        print(efficientgrasp_prediction.layers[-1].output_shape)
    # Return Model
    return efficientgrasp_train, efficientgrasp_prediction, all_layers



def build_BiFPN(backbone_feature_maps, bifpn_depth, bifpn_width, num_groups_gn, freeze_bn):
    """
    Building the bidirectional feature pyramid as described in https://arxiv.org/abs/1911.09070
    Args:
        backbone_feature_maps: Sequence containing the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        bifpn_depth: Number of BiFPN layer
        bifpn_width: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       fpn_feature_maps: Sequence of BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    fpn_feature_maps = backbone_feature_maps
    for i in range(bifpn_depth):
        fpn_feature_maps = build_BiFPN_layer(fpn_feature_maps, bifpn_width, num_groups_gn, i, freeze_bn = freeze_bn)
        
    return fpn_feature_maps

def build_BiFPN_layer(features, num_channels, num_groups_gn, idx_BiFPN_layer, freeze_bn = False):
    """
    Builds a single layer of the bidirectional feature pyramid
    Args:
        features: Sequence containing the feature maps of the previous BiFPN layer (P3, P4, P5, P6, P7) or the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    if idx_BiFPN_layer == 0:
        _, _, C3, C4, C5 = features
        P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in = prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, num_groups_gn, freeze_bn)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        
    #top down pathway
    input_feature_maps_top_down = [P7_in,
                                   P6_in,
                                   P5_in_1 if idx_BiFPN_layer == 0 else P5_in,
                                   P4_in_1 if idx_BiFPN_layer == 0 else P4_in,
                                   P3_in]
    
    P7_in, P6_td, P5_td, P4_td, P3_out = top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer, num_groups_gn)
    
    #bottom up pathway
    input_feature_maps_bottom_up = [[P3_out],
                                    [P4_in_2 if idx_BiFPN_layer == 0 else P4_in, P4_td],
                                    [P5_in_2 if idx_BiFPN_layer == 0 else P5_in, P5_td],
                                    [P6_in, P6_td],
                                    [P7_in]]
    
    P3_out, P4_out, P5_out, P6_out, P7_out = bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer, num_groups_gn)
    
    
    return P3_out, P4_td, P5_td, P6_td, P7_out #TODO check if it is a bug to return the top down feature maps instead of the output maps

def prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, num_groups_gn, freeze_bn):
    """
    Prepares the backbone feature maps for the first BiFPN layer
    Args:
        C3, C4, C5: The EfficientNet backbone feature maps of the different levels
        num_channels: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The prepared input feature maps for the first BiFPN layer
    """
    P3_in = C3
    P3_in = layers.Conv2D(num_channels, kernel_size = 1, padding = 'same', name = 'fpn_cells/cell_0/fnode3/resample_0_0_8/conv2d')(P3_in)
    P3_in = GroupNormalization(groups=num_groups_gn, axis=-1, epsilon = EPSILON, name='fpn_cells/cell_0/fnode3/resample_0_0_8/bn')(P3_in)
    
    P4_in = C4
    P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode2/resample_0_1_7/conv2d')(P4_in)
    P4_in_1 = GroupNormalization(groups=num_groups_gn, axis=-1, epsilon = EPSILON, name='fpn_cells/cell_0/fnode2/resample_0_1_7/bn')(P4_in_1)
    P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode4/resample_0_1_9/conv2d')(P4_in)
    P4_in_2 = GroupNormalization(groups=num_groups_gn, axis=-1, epsilon = EPSILON, name='fpn_cells/cell_0/fnode4/resample_0_1_9/bn')(P4_in_2)
    
    P5_in = C5
    P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode1/resample_0_2_6/conv2d')(P5_in)
    P5_in_1 = GroupNormalization(groups=num_groups_gn, axis=-1, epsilon = EPSILON, name='fpn_cells/cell_0/fnode1/resample_0_2_6/bn')(P5_in_1)
    P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode5/resample_0_2_10/conv2d')(P5_in)
    P5_in_2 = GroupNormalization(groups=num_groups_gn, axis=-1, epsilon = EPSILON, name='fpn_cells/cell_0/fnode5/resample_0_2_10/bn')(P5_in_2)
    
    P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
    P6_in = GroupNormalization(groups=num_groups_gn, axis=-1, epsilon = EPSILON, name='resample_p6/bn')(P6_in)
    P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
    
    P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
    
    return P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in

def top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer, num_groups_gn=3):
    """
    Computes the top-down-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing the input feature maps of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the top-down-pathway
    """
    feature_map_P7 = input_feature_maps_top_down[0]
    output_top_down_feature_maps = [feature_map_P7]
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(feature_map_other_level = output_top_down_feature_maps[-1],
                                                    feature_maps_current_level = [input_feature_maps_top_down[level]],
                                                    upsampling = True,
                                                    num_channels = num_channels,
                                                    idx_BiFPN_layer = idx_BiFPN_layer,
                                                    node_idx = level - 1,
                                                    op_idx = 4 + level,
                                                    num_groups_gn=num_groups_gn)
        
        output_top_down_feature_maps.append(merged_feature_map)
        
    return output_top_down_feature_maps

def bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer, num_groups_gn=3):
    """
    Computes the bottom-up-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing a list of feature maps serving as input for each level of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the bottom-up-pathway
    """
    feature_map_P3 = input_feature_maps_bottom_up[0][0]
    output_bottom_up_feature_maps = [feature_map_P3]
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(feature_map_other_level = output_bottom_up_feature_maps[-1],
                                                    feature_maps_current_level = input_feature_maps_bottom_up[level],
                                                    upsampling = False,
                                                    num_channels = num_channels,
                                                    idx_BiFPN_layer = idx_BiFPN_layer,
                                                    node_idx = 3 + level,
                                                    op_idx = 8 + level,
                                                    num_groups_gn=num_groups_gn)
        
        output_bottom_up_feature_maps.append(merged_feature_map)
        
    return output_bottom_up_feature_maps

def single_BiFPN_merge_step(feature_map_other_level, feature_maps_current_level, upsampling, num_channels, idx_BiFPN_layer, node_idx, op_idx, num_groups_gn=3):
    """
    Merges two feature maps of different levels in the BiFPN
    Args:
        feature_map_other_level: Input feature map of a different level. Needs to be resized before merging.
        feature_maps_current_level: Input feature map of the current level
        upsampling: Boolean indicating wheter to upsample or downsample the feature map of the different level to match the shape of the current level
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        node_idx, op_idx: Integers needed to set the correct layer names
    
    Returns:
       The merged feature map
    """
    if upsampling:
        feature_map_resampled = layers.UpSampling2D()(feature_map_other_level)
    else:
        feature_map_resampled = layers.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(feature_map_other_level)
    
    merged_feature_map = wBiFPNAdd(name = f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/add')(feature_maps_current_level + [feature_map_resampled])
    merged_feature_map = layers.Activation(lambda x: tf.nn.swish(x))(merged_feature_map)
    merged_feature_map = SeparableConvBlock(num_channels = num_channels,
                                            kernel_size = 3,
                                            strides = 1,
                                            name = f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/op_after_combine{op_idx}',
                                            num_groups_gn=num_groups_gn)(merged_feature_map)

    return merged_feature_map

def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn = False, num_groups_gn=3):
    """
    Builds a small block consisting of a depthwise separable convolution layer and a batch norm layer
    Args:
        num_channels: Number of channels used in the BiFPN
        kernel_size: Kernel site of the depthwise separable convolution layer
        strides: Stride of the depthwise separable convolution layer
        name: Name of the block
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The depthwise separable convolution block
    """
    f1 = layers.SeparableConv2D(num_channels, kernel_size = kernel_size, strides = strides, padding = 'same', use_bias = True, name = f'{name}/conv')
    f2 = GroupNormalization(groups=num_groups_gn, axis=-1, epsilon = EPSILON, name = f'{name}/bn')
    # return reduce(lambda f, g: lambda *args, **kwargs: f(*args, **kwargs), (f1, f2))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

class AngleNet(models.Model):
    def __init__(self, width, depth, num_classes = 12, num_anchors = 9, freeze_bn = False, **kwargs):
        super(AngleNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = self.width, bias_initializer = 'zeros', name = f'{self.name}/class-{i}', **options) for i in range(self.depth)]
        self.head = layers.SeparableConv2D(filters = self.num_classes * self.num_anchors, bias_initializer = PriorProbability(probability = 0.01), name = f'{self.name}/class-predict', **options)

        self.bns = [[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, self.num_classes))
        self.activation_sigmoid = layers.Activation('sigmoid')
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            # feature = self.bns[i][self.level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation_sigmoid(outputs)
        self.level += 1
        return outputs

class GraspNet(models.Model):
    def __init__(self, width, depth, num_iteration_steps, use_group_norm = True, num_groups_gn = 3, num_anchors = 1, freeze_bn = False, **kwargs):
        super(GraspNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.num_values = 4 # y, x, h, w
        # self.num_values = 6 # y, x, sin_t, cos_t, h, w
        # self.num_values = 5 # x, y, tan_t, h, w
        channel_axis=-1
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = self.width, name = f'{self.name}/box-{i}', **options) for i in range(self.depth)]
        self.bns = [[GroupNormalization(groups=num_groups_gn, axis=-1, epsilon = EPSILON, name = f'{self.name}/box-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        
        self.initial_grasp = layers.SeparableConv2D(filters = self.num_anchors * self.num_values, name = f'{self.name}/grasp-init-predict', **options)
    

        self.iterative_submodel = IterativeGraspSubNet(width = self.width,
                                                    depth = self.depth - 1,
                                                    num_values = self.num_values,
                                                    num_iteration_steps = self.num_iteration_steps,
                                                    num_anchors = self.num_anchors,
                                                    freeze_bn = freeze_bn,
                                                    use_group_norm = self.use_group_norm,
                                                    num_groups_gn = self.num_groups_gn,
                                                    name = "iterative_grasp_subnet")
        
        self.head = layers.SeparableConv2D(filters = self.num_anchors * self.num_values, name = f'{self.name}/grasp-predict', **options)
        self.reshape = layers.Reshape((-1, self.num_values))
        self.level = 0
        self.add = layers.Add()
        self.concat = layers.Concatenate(axis = channel_axis)

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            # feature = self.bns[i][self.level](feature)
            feature = self.activation(feature)
        
        grasp = self.initial_grasp(feature)
        
        for i in range(self.num_iteration_steps):
            iterative_input = self.concat([feature, grasp])
            delta_grasp = self.iterative_submodel([iterative_input, level], level_py = self.level, iter_step_py = i)
            grasp = self.add([grasp, delta_grasp])
        outputs = grasp
        # outputs = self.head(feature)
        outputs = self.reshape(outputs)
        self.level += 1
        return outputs

class IterativeGraspSubNet(models.Model):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = 3, **kwargs):
        super(IterativeGraspSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = width, name = f'{self.name}/iterative-grasp-sub-{i}', **options) for i in range(self.depth)]
        self.head = layers.SeparableConv2D(filters = self.num_anchors * self.num_values, name = f'{self.name}/iterative-grasp-sub-predict', **options)
        
        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/iterative-grasp-sub-{k}-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
        else: 
            self.norm_layer = [[[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/iterative-grasp-sub-{k}-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]

        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))

    def call(self, inputs, **kwargs):
        feature, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        
        return outputs


# build_EfficientGrasp_multi(0, print_architecture=True)