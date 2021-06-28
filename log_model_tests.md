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

302. 2021_06_12_16_37_53
Multi model
Normalization modified <-> Score Loss Scale Modified
Weight initialization modified
Subnet width/depth modified

WEIGHTS NORMALIZATION STDDEV INCREASE --> MORE SPREAD OF DATA
INPUT NORMALIZAION STDDEV INCREASE --> MORE SPREAD OF DATA

303. 2021_06_18_23_33_39
Multi model, Cornell
Only grasp loss, no score loss
multi model params modified

# Anchor Series
1. N1: 2021_06_22_20_57_00
    loss ratio: 1:1
    lr schedule: patience 5 cooldown 5
    lr: 1e-4
    epochs: 100
    IoU still low, angle decreasing good enough

2. N2: 2021_06_23_01_18_07
    loss ratio: 3:1
    lr schedule: patience 5 cd 0
    lr: 1e-4
    epochs: 100
    IoU STILL LOW, angle diff is good enough
        Total detected grasps:  305
        grasp_acc_img: 0.46
        grasp_accuracy: 0.3967
        avg_iou: 0.11
        avg_angle_diff: 11.64
        avg_pred_grasp_score: 0.59

3. N3: 2021_06_23_20_33_49
    RGD IMAGES: train_1, valid_1
    grasp bbox snapping to closest horizontal box by angle
    Total detected grasps:  324
    grasp_acc_img: 0.62
    grasp_accuracy: 0.7130
    avg_iou: 0.22
    avg_angle_diff: 9.81
    avg_pred_grasp_score: 0.58
    --IoU decent now. Score threshold needs to be reduced to 0.3 from 0.5 to get more detections and better accuracy
    ----------- Results ----------------------------------
    Images to check:  [70, 80, 92, 97, 113, 118, 132]
    grasp_acc_img: 0.99
    top_score_accuracy: 0.9605
    grasp_accuracy: 0.8906
    avg_iou: 0.17
    avg_angle_diff: 16.15
    avg_pred_grasp_score: 0.46
    --------Results including top k ----------------------
    grasp_acc_img: 0.99
    top_score_accuracy: 0.9605
    top_k_accuracy: 0.9887 for k = 5
    grasp_accuracy: 0.9201
    avg_iou: 0.20
    avg_angle_diff: 15.13
    avg_pred_grasp_score: 0.48

4. N4: 2021_06_24_02_57_59
    On VRMD Dataset
    LR: 1e-5
    50 epochs
    ----------- Results ----------------------------------
    Images to check:  [1, 6, 8, 9, 20, 21, 25, 27, 40, 48, 50, 56, 59, 61, 65, 69, 71, 73, 75, 76, 77, 80, 81, 82, 89, 91, 92, 97, 105, 110, 115, 116, 118, 119, 127, 131, 133, 140, 142, 157, 159, 160, 161, 165, 169, 172, 177, 181, 192, 195, 196, 200, 217, 218, 222, 223, 232, 245, 247, 251, 252, 268, 304, 312, 325, 326, 353, 364, 365, 366, 369, 388, 408, 414, 424, 427, 429, 431, 436, 443, 467, 474, 476, 511, 572, 614, 633, 657, 658, 669, 698, 745, 818, 821, 851, 853, 854, 906, 916, 946, 956, 957, 969, 971, 1048, 1075, 1088, 1092, 1095, 1106, 1107, 1112, 1114, 1118, 1123, 1128, 1129, 1133, 1142, 1145, 1147, 1149, 1154, 1161, 1172, 1175, 1190, 1198, 1200, 1201, 1202, 1204, 1212, 1235, 1236, 1239, 1240, 1241, 1245, 1246, 1252, 1257, 1259, 1264, 1265, 1270, 1272, 1277, 1281, 1283, 1284, 1287, 1293, 1294, 1296, 1299, 1302, 1305, 1308, 1309, 1316, 1319, 1320, 1324, 1328, 1336, 1339, 1341, 1342, 1345, 1349, 1356, 1361, 1368, 1376, 1381, 1384, 1390, 1392, 1393, 1396, 1398, 1400, 1401, 1406, 1416, 1417, 1419, 1421, 1425, 1426, 1437, 1440, 1447, 1459, 1460, 1461, 1462, 1471, 1480, 1491, 1494, 1519, 1534, 1541, 1547, 1548, 1559, 1566, 1567, 1569, 1571, 1577, 1580, 1582, 1583, 1584, 1586, 1595, 1606, 1608, 1609, 1614, 1615, 1622, 1624, 1629, 1630, 1634, 1640, 1643, 1652, 1657, 1661, 1670, 1672, 1675, 1678, 1679, 1680, 1682, 1684, 1685, 1691, 1693, 1694, 1702, 1703, 1704, 1708, 1717, 1727, 1747, 1774, 1779, 1788, 1793, 1807, 1817, 1819, 1831, 1837, 1855, 1869, 1870, 1871, 1872, 1876, 1878, 1880, 1948, 1959, 1966, 1975, 1988, 1993, 1997, 2010, 2012, 2030, 2036, 2046, 2056, 2073, 2088, 2100, 2104, 2126, 2127, 2128, 2132, 2135, 2136, 2144, 2145, 2146, 2147, 2158, 2159, 2168, 2173, 2178, 2179, 2182, 2188, 2190, 2194, 2201, 2202, 2203, 2214, 2244, 2262, 2280, 2281, 2286, 2292, 2294, 2297, 2298, 2299, 2302, 2304, 2305, 2306, 2307, 2311, 2335, 2348, 2350, 2351, 2357, 2360, 2361, 2362, 2367, 2378, 2383, 2386, 2401, 2416, 2420, 2425, 2441, 2443, 2449, 2462, 2474, 2506, 2507, 2518, 2519, 2558, 2607, 2610, 2612, 2643, 2650, 2652, 2656, 2661, 2666, 2681, 2692, 2695, 2696, 2712, 2736, 2741, 2743, 2744, 2746, 2764, 2769, 2775, 2783, 2822, 2824, 2825, 2826, 2827, 2837, 2839, 2847, 2855, 2859, 2860, 2862, 2865, 2866, 2868, 2871, 2873, 2874, 2875, 2888, 2893, 2894, 2895, 2896, 2900, 2912, 2916, 2917, 2921, 2927, 2929, 2937, 2938, 2947, 2952, 2961, 2964, 2969, 2972, 2977, 2981, 2985, 2988, 2989, 2993, 3000, 3002, 3006, 3013, 3014, 3017, 3018, 3022, 3024, 3025, 3027, 3035, 3040, 3058, 3074, 3094, 3095, 3101, 3106, 3107, 3111, 3115, 3127, 3133, 3166, 3169, 3170, 3181, 3189, 3203, 3226, 3229, 3249, 3276, 3284, 3287, 3322, 3325, 3337, 3338, 3343, 3355, 3357, 3362, 3364, 3367, 3383, 3386, 3392, 3393, 3411, 3421, 3434, 3445, 3461, 3465, 3470, 3472, 3481, 3487, 3495, 3510, 3521, 3559]
    grasp_acc_img: 0.97
    top_score_accuracy: 0.8464
    grasp_accuracy: 0.8460
    avg_iou: 0.02
    avg_angle_diff: 16.54
    avg_pred_grasp_score: 0.37

5. N3b: 2021_06_25_18_19_28
CORNELL RGB IMAGES
    LR: 1e-4, patience 5, cd 0, 150 epochs
    loss: 0.8085 - angle_classification_loss: 0.3031 - bbox_regression_loss: 0.1685 - val_loss: 1.3414 - val_angle_classification_loss: 0.4644 - val_bbox_regression_loss: 0.2923
    Total detected grasps:  1019
    grasp_acc_img: 0.97
    top_score_accuracy_img: 0.75
    grasp_accuracy: 0.6222
    avg_iou: 0.16
    avg_angle_diff: 16.25
    avg_pred_grasp_score: 0.46
    ------
Inference Code Results
    Images to check:  [50, 59, 72, 84, 89, 105, 113, 118, 148, 162]
    grasp_acc_img: 0.99
    top_score_accuracy: 0.9435
    top_k_accuracy: 0.9887 for k = 5
    grasp_accuracy: 0.9256
  (Below calculated for top k grasps)
    avg_iou: 0.20
    avg_angle_diff: 14.55
    avg_pred_grasp_score: 0.49

6. N5. Amazon: 2021_06_27_22_45_12
    --- inference best val loss ---
    Images to check:  [9, 16, 21, 24, 31, 33, 42, 44, 48, 58, 60, 63]
    grasp_acc_img: 0.68
    top_score_accuracy: 0.6176
    top_k_accuracy: 0.6765 for k = 5
    grasp_accuracy: 0.7966
    avg_iou: 0.51
    avg_angle_diff: 2.85
    avg_pred_grasp_score: 0.42
    --- (CHOSEN) inference finish ---
    Images to check:  [14, 16, 20, 21, 23, 24, 25, 27, 28, 29, 31, 33, 35, 42, 43, 53, 57, 59, 63]
    grasp_acc_img: 0.87
    top_score_accuracy: 0.6765
    top_k_accuracy: 0.8382 for k = 5
    grasp_accuracy: 0.6480
    avg_iou: 0.44
    avg_angle_diff: 6.22
    avg_pred_grasp_score: 0.58

7. N3c: Cornell: 
    LIGHTER MODEL
# A Series:
RGB Images
bifpn_widths = (144, 60, 96)   # 144
bifpn_depths = (3, 3, 3)                # 3
subnet_depths = (3, 3, 4)               # 4
subnet_width = (96, 36, 48)    # 96
subnet_iteration_steps = (2, 1, 2)      # 2
num_groups_gn = (6, 3, 3)           # 6  #try to get 16 channels per group ## width > groups * 16 