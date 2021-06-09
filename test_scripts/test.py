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
import numpy as np
idx = np.round(np.linspace(0, 20 - 1, 5)).astype(int)
print(idx)