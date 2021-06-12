fname = "/home/aby/Workspace/parallel-jaw-grasping-dataset/data/label/000000.good.txt"
grs = []
# jaw_size = 20
# with open(fname) as f:
#     while True:
#         # Load 4 lines at a time, corners of bounding box.
#         line = f.readline()
#         if not line:
#             break  # EOF
#         end_coords = line.split()
#         end_coords = [int(coords) for coords in end_coords]
#         x0, y0, x1, y1 = end_coords
#         x_diff = x1-x0
#         y_diff = y1-y0
#         norm_val = (x_diff**2 + y_diff**2)**0.5
#         dir_vec = [y_diff/norm_val, -x_diff/norm_val]
#         p0 = [ int(y0+dir_vec[1]*jaw_size/2), int(x0+dir_vec[0]*jaw_size/2) ]
#         p1 = [ int(y0-dir_vec[1]*jaw_size/2), int(x0-dir_vec[0]*jaw_size/2) ]
#         p2 = [ int(y1-dir_vec[1]*jaw_size/2), int(x1-dir_vec[0]*jaw_size/2) ]
#         p3 = [ int(y1+dir_vec[1]*jaw_size/2), int(x1+dir_vec[0]*jaw_size/2) ]

#         try:
#             gr = np.array([
#                 p0,
#                 p1,
#                 p2,
#                 p3
#             ])

#             grs.append(GraspRectangle(gr))

#         except ValueError:
#             # Some files contain weird values.
#             continue
# import numpy as np
# arr = np.array([[4, 1], [2, 3], [1, 2], [0, 4]])
# print(arr)
# # arr=np.sort(arr, axis=1)
# sortedArr = arr[(-arr[:,1]).argsort()]
# print(sortedArr)

# import matplotlib.pyplot as plt
# import numpy as np
# import mpld3

# fig, ax = plt.subplots()
# N = 100

# scatter = ax.scatter(np.random.normal(size=N),
#                      np.random.normal(size=N),
#                      c=np.random.random(size=N),
#                      s=1000 * np.random.random(size=N),
#                      alpha=0.3,
#                      cmap=plt.cm.jet)
# ax.grid(color='white', linestyle='solid')

# ax.set_title("Scatter Plot (with tooltips!)", size=20)

# labels = ['point {0}'.format(i + 1) for i in range(N)]
# tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
# mpld3.plugins.connect(fig, tooltip)

# mpld3.show()


import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from dataset_processing import grasp

gp = grasp.Grasp([10,11], 1, 0, 10, 20, 0.5)
print(gp.center)
norm_gp = gp.as_list
print(norm_gp)
unnorm_gp = grasp.Grasp(norm_gp[0:2], *norm_gp[2:], unnorm=True)
print(unnorm_gp.center)
