### Single Model

1. 2021_05_27_02_42_08

"--batch-size", "4",
"--lr", "1e-4",
"--epochs", "100",
Total params: 4,226,285
Trainable params: 4,186,829
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