# from dataset_processing.grasp import get_grasp_from_pred, Grasp
# import numpy as np
# # y, x, theta, l, w
# test_eg = [100.0, 200.0, np.pi/3, 10, 5]
# # test_eg = [100, 200, 0, 0, 0]
# t_grasp = Grasp(test_eg[0:2],  *test_eg[2:])
# ang_class = int(round( (t_grasp.angle+np.pi/2) / (np.pi / 12))) % 12
# print('A1: ', t_grasp.angle * 180/ np.pi)
# print('HB1: ', t_grasp.as_horizontal_bbox)
# print('Comp 1: ', t_grasp.as_bbox)
# print('Comp 1: ', t_grasp.as_gr.points)
# print('Angle Class: ', ang_class)

# p_grasp = get_grasp_from_pred(t_grasp.as_horizontal_bbox, int(round( (t_grasp.angle+np.pi/2) / (np.pi / 12))) % 12)
# ang_class2 = int(round( (p_grasp.angle+np.pi/2) / (np.pi / 12))) % 12
# print('A2: ', p_grasp.angle * 180/ np.pi)
# print('HB2: ', p_grasp.as_horizontal_bbox)
# print('Comp 2: ', p_grasp.as_bbox)
# print('Comp 2: ', p_grasp.as_gr.points)
# print('Angle Class: ', ang_class2)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# # ax.imshow(rgd_img)
# (t_grasp.as_gr).plot(ax, 1)
# (p_grasp.as_gr).plot(ax, 1)
# plt.show()

from dataset_processing.cornell_generator import CornellDataset
from dataset_processing.vmrd_generator import VMRDDataset
import json

# ### TESTING
# dataset = "/home/aby/Workspace/Cornell/archive"
# with open(dataset+'/valid_1.txt', 'r') as filehandle:
#     train_data = json.load(filehandle)

# train_generator = CornellDataset(
#     dataset,
#     train_data,
#     train=False,
#     shuffle=False,
#     batch_size=1
# )

# for i in range(0, 20):
#     # train_generator[i]
#     train_generator.get_annotation_val(i)

### VMRD TESTING
dataset = '/home/aby/Workspace/vmrd-v2'
with open(dataset+'/ImageSets/Main/trainval.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    data = []
    for line in lines:
        data.append(line.strip())
train_data = []
valid_data = []
for i in range(len(data)):
    if not i%10 == 0:
        train_data.append(data[i])
    else:
        valid_data.append(data[i])

train_generator = VMRDDataset(
    dataset,
    train_data,
    train=False,
    shuffle=False,
    batch_size=1
)

for i in range(0, 20):
    # train_generator[i]
    train_generator.get_annotation_val(i)