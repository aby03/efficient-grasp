from model import *

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
    grasp_net = GraspNet(subnet_width,
                        subnet_depth,
                        num_iteration_steps = subnet_num_iteration_steps,
                        freeze_bn=freeze_bn, 
                        use_group_norm = True,
                        num_groups_gn = num_groups_gn,
                        name='grasp_net')
    
    # Get reshape dims for applying grasp net
    reshape_dim = 0     # 64x64 + 32x32 + 16x16 + 8x8 + 4x4
    for i in range(len(fpn_feature_maps)):
        reshape_dim += fpn_feature_maps[i].shape[1]*fpn_feature_maps[i].shape[2]
    
    # Apply GraspNet
    grasp_regression = [grasp_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    grasp_regression = layers.Concatenate(axis=1, name='regression_con')(grasp_regression)
    grasp_regression = layers.Reshape((-1,reshape_dim * output_dim))(grasp_regression) # 5456 for num_anchors=1 && 49104 for 9 

    grasp_regression_multi = layers.Dense(pred_count * output_dim, name='regression_grasp'
                                        # , kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=256.0)
                                        , kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=50.0)
                                        )(grasp_regression)
    grasp_regression_multi = layers.Reshape((pred_count, output_dim), name='regression_grasp_re')(grasp_regression_multi)
    grasp_regression_score = layers.Dense(pred_count, name='regression_score', activation=tf.keras.activations.sigmoid)(grasp_regression)
    grasp_regression_score = layers.Reshape((pred_count,1), name='regression_score_re')(grasp_regression_score)

    grasp_regression_out = layers.Concatenate(axis=2, name='regression_out')([grasp_regression_multi, grasp_regression_score])

    # grasp_regression = layers.Dense(output_dim, name='regression_out')(grasp_regression)
    # # Reshape to (batch, 1, 6) and then Concatenate 30 times
    # grasp_regression_train = layers.Concatenate(axis=1, name="final_layer")([grasp_regression for x in range(0, 30)])

    # # Prediction is (batch, 6)
    # grasp_regression_pred = layers.Flatten(name='regression_p')(grasp_regression)

    # Build Complete Model
    efficientgrasp_train = models.Model(inputs = [image_input], outputs = grasp_regression_out, name = 'efficientgrasp')
    efficientgrasp_prediction = models.Model(inputs = [image_input], outputs = grasp_regression_out, name = 'efficientgrasp_prediction')


    # # Debug Architecture
    # grasp_regression = layers.MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='valid')(image_input)
    # grasp_regression = layers.Flatten()(grasp_regression)

    # # Final Layer into 6 dim vector
    # grasp_regression = layers.Dense(output_dim, name='regression')(grasp_regression)

    # # Reshape to (batch, 1, 6) and then Concatenate 30 times
    # grasp_regression = layers.Reshape(target_shape=(1,6))(grasp_regression)
    # grasp_regression = layers.Concatenate(axis=1, name="final_layer")([grasp_regression for x in range(0, 30)])

    # efficientgrasp_train = models.Model(inputs = [image_input], outputs = grasp_regression, name = 'efficientgrasp')
    
    all_layers = list(set(efficientgrasp_train.layers + efficientgrasp_prediction.layers))

    if print_architecture:
        efficientgrasp_train.summary()
    # Return Model
    return efficientgrasp_train, efficientgrasp_prediction, all_layers

# build_EfficientGrasp_multi(0, print_architecture=True)