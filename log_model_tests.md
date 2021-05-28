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

2. 2021_05_27_06_37_50 (TO RRUN)
"--batch-size", "4",
"--lr", "1e-4",
"--epochs", "100",
Total params: 4,020,965
Trainable params: 386,121
Non-trainable params: 3,634,844
Params
    bifpn_widths = (64, 96)   # 144
    bifpn_depths = (3, 3)                # 3
    subnet_depths = (3, 4)               # 4
    subnet_width = (36, 48)    # 96
    subnet_iteration_steps = (1, 2)      # 2
    num_groups_gn = (3, 3)           # 6  #try to get 16 channels per group


101. 2021_05_28_03_40_07
Multi model 100 epochs trained
To train for another 100 epochs