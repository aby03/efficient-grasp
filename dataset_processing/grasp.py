import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max

# Normalization constants
GRASP_MEAN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GRASP_STD = [1.0, 1.0, 0.02, 0.02, 1.0, 1.0]
# GRASP_MEAN = [241.8, 291.3, 0.1664, 0.1093, 27.33, 46.25]
# GRASP_STD = [26.88,  42.22,   0.7140,  0.6712, 11.29, 21.29]
# GRASP_MEAN = [256.0,   256.0,     0.0,  0.0, 60.0, 40.0]
# GRASP_STD = [5.0,  5.0,     0.2,  0.2, 5.0, 5.0]
y_std = GRASP_STD[0]
x_std = GRASP_STD[1]
sin_std = GRASP_STD[2]
cos_std = GRASP_STD[3]
h_std = GRASP_STD[4]
w_std = GRASP_STD[5]

y_mean = GRASP_MEAN[0]
x_mean = GRASP_MEAN[1]
sin_mean = GRASP_MEAN[2]
cos_mean = GRASP_MEAN[3]
h_mean = GRASP_MEAN[4]
w_mean = GRASP_MEAN[5]

def get_grasp_from_pred(pred_grasp_bbox, pred_grasp_angle_class):
    pred_grasp_angle = (np.pi / 12) * pred_grasp_angle_class - np.pi/2      # angle in radian
    # if not (pred_grasp_angle_class > 3 and pred_grasp_angle_class < 9):
    #     pred_grasp_angle = -( np.pi / 2 - pred_grasp_angle)
    #     pred_grasp = [
    #                     (pred_grasp_bbox[0] + pred_grasp_bbox[2]) / 2,      # x
    #                     (pred_grasp_bbox[1] + pred_grasp_bbox[3]) / 2,      # y
    #                     pred_grasp_angle,                                   # theta
    #                     (pred_grasp_bbox[3] - pred_grasp_bbox[1]),       # w
    #                     (pred_grasp_bbox[2] - pred_grasp_bbox[0])      # h
    #                 ]
    # else:
    pred_grasp = [
                    (pred_grasp_bbox[0] + pred_grasp_bbox[2]) / 2,      # x
                    (pred_grasp_bbox[1] + pred_grasp_bbox[3]) / 2,      # y
                    pred_grasp_angle,                                   # theta
                    (pred_grasp_bbox[2] - pred_grasp_bbox[0]),      # h
                    (pred_grasp_bbox[3] - pred_grasp_bbox[1])       # w
                ]
    # y, x, theta, h, w
    pred_grasp_obj = Grasp((pred_grasp[1], pred_grasp[0]), *pred_grasp[2:])
    return pred_grasp_obj

def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Hehe
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)
    
    @classmethod
    def load_from_vmrd_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            text = f.read()
            lines = text.split("\n")
            counter = 0
            for line in lines:
            #     if line:
            #         counter += 1
            # p_lines = lines
            # if counter > 100:
            #     p_lines = []
            #     idx = np.round(np.linspace(0, counter - 1, NO_OF_GRASPS_TO_PROCESS, dtype='int')).astype(int)
            #     # print('.......TE.....', type(idx))
            #     for i in range(0, idx.shape[0]):
            #         p_lines.append( lines[idx[i]] )
            # # while True:
            # for line in p_lines:
                if not line:
                    break  # EOF
                box_coords = line.split()
                box_coords = [float(coords) for coords in box_coords[:-2]]
                x0, y0, x1, y1, x2, y2, x3, y3 = box_coords
                p0 = [ y0, x0 ]
                p1 = [ y1, x1 ]
                p2 = [ y2, x2 ]
                p3 = [ y3, x3 ]
                try:
                    gr = np.array([
                        p0,
                        p1,
                        p2,
                        p3
                    ])
                    grs.append(GraspRectangle(gr))
                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    @classmethod
    def load_from_amazon_file(cls, fname):
        """
        Load grasp rectangles from a Amazon 2017 dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        NO_OF_GRASPS_TO_PROCESS = 100
        grs = []
        jaw_size = 20
        with open(fname) as f:
            text = f.read()
            lines = text.split("\n")
            counter = 0
            for line in lines:
                if line:
                    counter += 1
            p_lines = lines
            if counter > 100:
                p_lines = []
                idx = np.round(np.linspace(0, counter - 1, NO_OF_GRASPS_TO_PROCESS, dtype='int')).astype(int)
                # print('.......TE.....', type(idx))
                for i in range(0, idx.shape[0]):
                    p_lines.append( lines[idx[i]] )
            # while True:
            for line in p_lines:
                if not line:
                    break  # EOF
                end_coords = line.split()
                end_coords = [int(coords) for coords in end_coords]
                x0, y0, x1, y1 = end_coords
                x_diff = x1-x0
                y_diff = y1-y0
                norm_val = (x_diff**2 + y_diff**2)**0.5
                dir_vec = [y_diff/norm_val, -x_diff/norm_val]
                p0 = [ int(y0+dir_vec[1]*jaw_size/2), int(x0+dir_vec[0]*jaw_size/2) ]
                p1 = [ int(y0-dir_vec[1]*jaw_size/2), int(x0-dir_vec[0]*jaw_size/2) ]
                p2 = [ int(y1-dir_vec[1]*jaw_size/2), int(x1-dir_vec[0]*jaw_size/2) ]
                p3 = [ int(y1+dir_vec[1]*jaw_size/2), int(x1+dir_vec[0]*jaw_size/2) ]
                try:
                    gr = np.array([
                        p0,
                        p1,
                        p2,
                        p3
                    ])
                    grs.append(GraspRectangle(gr))
                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                # grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
                grs.append(Grasp(np.array([y, x]), np.sin(-theta), np.cos(-theta), w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    self.points: 2d numpy array (4,2)
    [0, 0] is 1st point y, [0, 1] is 1nd point x
    """

    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_five_vec(self):
        '''
        center: [y, x] format
        '''
        return [*self.center, self.angle, self.length, self.width]

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.float32)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, np.sin(self.angle), np.cos(self.angle), length=self.length / 3, width=self.width).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """
        if abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def corner_scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        Scale and points are in (y,x) format
        """
        if factor == (1.0, 1.0):
            return
        self.points[:,0] = np.round_(self.points[:,0] * factor[0]).astype(int)
        self.points[:,1] = np.round_(self.points[:,1] * factor[1]).astype(int)

    def plot(self, ax, q=0.0, color='green'):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param q: Grasp quality
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)
        ax.plot(self.center[1], self.center[0], 'o', color=color, label='_nolegend_')
        # ax.legend(['Red box: Predicted Grasp', 'Green box: Labelled Grasps'])
        # ax.legend(['score: {0:.2f}'.format(q)])

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        if (factor == 1.0):
            return
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """

    def __init__(self, center, angle, length, width, quality=-1, unnorm=False):
        # Assign Values
        self.center = center    # in [y, x] format

        self.angle = angle
        self.quality = quality
        self.length = float(length)
        self.width = float(width)

        # ### Make Rotation less than 45 degree
        if self.angle > np.pi / 4 or self.angle < - np.pi / 4:
            self.angle = -( np.pi / 2 - self.angle)
            # Swap L, W
            self.length = float(width)
            self.width = float(length)

        self.sin_t = np.sin(self.angle)
        self.cos_t = np.cos(self.angle)
        # print('F: C-{} A-{} L-{} W-{} S-{} C-{}'.format(self.center, self.angle, self.length, self.width, self.sin_t, self.cos_t))



    def unnormalize(self, center, sin_t, cos_t, h, w):
        center[0] = center[0] * y_std + y_mean
        center[1] = center[1] * x_std + x_mean
        sin_t = sin_t * sin_std + sin_mean
        cos_t = cos_t * cos_std + cos_mean
        h = h * h_std + h_mean
        w = w * w_std + w_mean
        return center, sin_t, cos_t, h, w

    def normalize(self, center, sin_t, cos_t, h, w):
        center[0] = (center[0] - y_mean) / y_std
        center[1] = (center[1] - x_mean) / x_std
        sin_t = (sin_t - sin_mean) / sin_std
        cos_t = (cos_t - cos_mean) / cos_std
        h = (h - h_mean) / h_std
        w = (w - w_mean) / w_std
        return center, sin_t, cos_t, h, w

    @property
    def as_angle(self):
        return self.angle

    @property
    def as_list(self):
        center, sin_t, cos_t, length, width = self.normalize(self.center, self.sin_t, self.cos_t, self.length, self.width)
        return [*center, 
                sin_t,
                cos_t, 
                length, 
                width]
    @property
    def as_bbox(self):
        """
        Convert to list of bboxes
        :return: list of points in [x,y] format
        """
        xo = self.cos_t
        yo = self.sin_t

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return  [
                    [x1 - self.width / 2 * yo, y1 - self.width / 2 * xo],
                    [x2 - self.width / 2 * yo, y2 - self.width / 2 * xo],
                    [x2 + self.width / 2 * yo, y2 + self.width / 2 * xo],
                    [x1 + self.width / 2 * yo, y1 + self.width / 2 * xo],
                ]
    
    @property
    def as_horizontal_bbox(self):
        """
        Convert to list of bboxes
        :return: list of points in [x1(min), y1(min), x2(max), y2(max)]
        """
        # if not abs(self.angle) <= np.pi/4:
        # xo = 1 #self.cos_t
        # yo = 0 #self.sin_t
        # else:
        #     xo = 0 #self.cos_t
        #     yo = 1 #self.sin_t

        # y1 = self.center[0] + self.length / 2 * yo
        # x1 = self.center[1] - self.length / 2 * xo
        # y2 = self.center[0] - self.length / 2 * yo
        # x2 = self.center[1] + self.length / 2 * xo
        # return [
        #         x1 - self.width / 2 * yo, y1 - self.width / 2 * xo,
        #         x2 + self.width / 2 * yo, y2 + self.width / 2 * xo,
        #        ]
        return  [
                    self.center[1] - self.length / 2, self.center[0] - self.width / 2,
                    self.center[1] + self.length / 2, self.center[0] + self.width / 2,
                ]

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = self.cos_t
        yo = self.sin_t

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(np.array(
            [
                [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],
                [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
                [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
                [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
            ]
        ).astype(np.float))

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, self.quality, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        angle = np.arctan(self.sin_t/self.cos_t)
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[1] * scale, self.center[0] * scale, -1 * angle * 180 / np.pi, self.length * scale,
            self.width * scale)

# Not used, need to update
def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=10, threshold_abs=0.02, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]
        grasp_quality = q_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle, quality=grasp_quality)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasps