### Single Model

1. 2021_05_27_02_41_14   2021_05_27_02_42_08

"--batch-size", "4",
"--lr", "1e-4",
"--epochs", "100",
Total params: 4,226,285
Trainable params: 591,441 (RECHECK)
Non-trainable params: 39,456
Params
    bifpn_widths = (96)
    bifpn_depths = (3)
    subnet_depths = (4)
    subnet_width = (48)
    subnet_iteration_steps = (2)
    num_groups_gn = (3)
callbacks.append(keras.callbacks.ReduceLROnPlateau(
    monitor    = 'loss',
    factor     = 0.5,
    patience   = 5,
    verbose    = 1,
    mode       = 'min',
    min_delta  = 0.0001,
    cooldown   = 0,
    min_lr     = 1e-6
))
> Results
-LR reduced at 49, 57, 77, 82, 91, 96 epochs
loss: 62.5132
val_grasp_loss: 100.64
grasp_accuracy: 0.8870
avg_iou: 0.52
avg_angle_diff: 7.30
Best Accuracy: 90.395% at 81 epoch
Training Time: 2hrs40mins
Memory Usage:
Computation Time:

2. 2021_06_04_23_13_18
"--batch-size", "4",
"--lr", "1e-4",
"--epochs", "100",
Total params: 4,020,965
Trainable params: 386,121
Non-trainable params: 3,634,844
Params
    bifpn_widths = (60, 96)   # 144
    bifpn_depths = (3, 3)                # 3
    subnet_depths = (3, 4)               # 4
    subnet_width = (36, 48)    # 96
    subnet_iteration_steps = (1, 2)      # 2
    num_groups_gn = (3, 3)           # 6  #try to get 16 channels per group
Training Duration: 2:14:17.621749
light model
train_0.txt

3. 2021_06_05_01_37_09
"--batch-size", "4",
"--lr", "1e-4",
"--epochs", "100",
Total params: 4,020,965
Trainable params: 386,121
Non-trainable params: 3,634,844
Params
    bifpn_widths = (60, 96)   # 144
    bifpn_depths = (3, 3)                # 3
    subnet_depths = (3, 4)               # 4
    subnet_width = (36, 48)    # 96
    subnet_iteration_steps = (1, 2)      # 2
    num_groups_gn = (3, 3)           # 6  #try to get 16 channels per group
Training Duration: 2:14:17.621749
light model
train_1.txt
Training Duration: 2:15:22.868625
> Results
    Acc: Best-85.31%(49 epoch)
    loss: 68.4413
    val_grasp_loss: 99.92
    grasp_accuracy: 0.8136
    avg_iou: 0.48
    avg_angle_diff: 8.92


101. 2021_05_28_03_40_07
Multi model 100 epochs trained
To train for another 100 epochs

102. 2021_05_28_22_24_04
Another 100 epochs trained starting with 1e-5 LR

201. 2021_06_05_23_27_08
Amazon Dataset
Trained for 200 epochs
52% acc

202. 2021_06_07_03_48_48
Amazon Dataset
Previous Trained for another 200 epochs (from 201-400 epochs)
58% accuracy

203. 2021_06_11_06_48_40
Amazon dataset

301. 2021_06_10_03_38_02
Cornell Dataset
Multi model score loss scaled by 1000 
Best accuracy model 97% is of unstable values(DONT USE)
Pick Final model
Loss oscillation stabilized after LR reduced to 3e-6
Final accuracy: 90.4

302.
Multi model
Normalization modified <-> Score Loss Scale Modified
Weight initialization modified
Subnet width/depth modified

WEIGHTS NORMALIZATION STDDEV INCREASE --> MORE SPREAD OF DATA
INPUT NORMALIZAION STDDEV INCREASE --> MORE SPREAD OF DATA