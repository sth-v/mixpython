import numpy as np
from numpy import ndarray
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from functools import wraps


def __curry__(func):
    """
    >>> @__curry__
    ... def foo(a, b, c):
    ...     return a + b + c
    >>> foo(1)
    <function __main__.foo>
    """

    @wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)

        @wraps(func)
        def new_curried(*args2, **kwargs2):
            return curried(*(args + args2), **dict(kwargs, **kwargs2))

        return new_curried

    return curried


def point_line_side(points, line_start, line_end):
    left_side = []
    right_side = []
    on_side = []

    line_start_x = line_start[0]
    line_start_y = line_start[1]

    line_end_x = line_end[0]
    line_end_y = line_end[1]
    x_new = points[:, 0]
    y_new = points[:, 1]

    for i in range(np.alen(x_new)):
        d = (
                (x_new[i] -
                 line_start_x) * (line_end_y -
                                  line_start_y) - (y_new[i] -
                                                   line_start_y) * (line_end_x -
                                                                    line_start_x))
        if d > 0:
            left_side.append([x_new[i], y_new[i]])
        if d == 0:
            on_side.append([x_new[i], y_new[i]])
        if d < 0:
            right_side.append([x_new[i], y_new[i]])

    return np.array([left_side, right_side, on_side])


def convexhull(points: ndarray):
    hull = ConvexHull(points)
    sorted = []
    for i in hull.vertices:
        sorted.append(points[i])
    return sorted


def minimum_bound_rectangle(points: ndarray):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    :param points: a nx2 matrix of coordinates
    :rval: a nx2 matrix of coordinates
    """

    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def get_clusters(data: ndarray, labels):
    """
    :param data: The dataset
    :param labels: The label for each point in the dataset
    :return: List[np.ndarray]: A list of arrays where the elements of each array
    are data points belonging to the label at that ind
    """
    return [data[np.where(labels == i)] for i in range(np.amax(labels) + 1)]


class SortedKMean(object):
    """
    >>> points = np.array([[[661.0, 249.0], [750.0, 274.0], [635.0, 276.0]],
    ...            [[706.0, 355.0], [635.0, 276.0], [750.0, 274.0]],
    ...            [[706.0, 355.0], [750.0, 274.0], [778.0, 334.0]],
    ...            [[778.0, 334.0], [672.0, 398.0], [706.0, 355.0]],
    ...            [[606.0, 492.0], [672.0, 398.0], [747.0, 511.0]],
    ...            [[598.0, 428.0], [672.0, 398.0], [606.0, 492.0]],
    ...            [[778.0, 334.0], [747.0, 511.0], [672.0, 398.0]]], dtype = np.float64).reshape((21,2))
    >>> points
    array([[661., 249.],
           [750., 274.],
           [635., 276.],
           [706., 355.],
           [635., 276.],
           [750., 274.],
           [706., 355.],
           [750., 274.],
           [778., 334.],
           [778., 334.],
           [672., 398.],
           [706., 355.],
           [606., 492.],
           [672., 398.],
           [747., 511.],
           [598., 428.],
           [672., 398.],
           [606., 492.],
           [778., 334.],
           [747., 511.],
           [672., 398.]])
    >>> km = SortedKMean(points)
    <__main__.SortedKMean at 0x24fd8dda6d0>
    >>> def custom_solver(points, kmean, x):
    ...     labels =  kmean().labels_
    ...     return [points[np.where(labels == i)] * x for i in range(np.amax(labels) + 1)]
    <function __main__.custom_solver(points, kmean, x)>
    >>> km.calculate = custom_solver
    ... km.calculate()
    <function __main__.custom_solver(points, kmean, x)>
    >>> km.calculate(x=3)
    [array([[2016., 1194.],
            [1818., 1476.],
            [2016., 1194.],
            [2241., 1533.],
            [1794., 1284.],
            [2016., 1194.],
            [1818., 1476.],
            [2241., 1533.],
            [2016., 1194.]]),
     array([[1983.,  747.],
            [2250.,  822.],
            [1905.,  828.],
            [2118., 1065.],
            [1905.,  828.],
            [2250.,  822.],
            [2118., 1065.],
            [2250.,  822.],
            [2334., 1002.],
            [2334., 1002.],
            [2118., 1065.],
            [2334., 1002.]])]
    """

    def __init__(self, points, **kwargs):
        self._points = points
        self._n_clusters = 2
        self._random_state = 0
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.kmean = lambda **x: KMeans(self._n_clusters, **dict({'random_state': self._random_state}, **x)).fit(
            self.points)
        self._calculate = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, v):
        self._points = v

    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, v):
        self._n_clusters = v

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, v):
        self._random_state = v

    @property
    def calculate(self):
        return self._calculate

    @calculate.setter
    def calculate(self, func):
        kwarg = {}
        for k in func.__code__.co_varnames:
            if hasattr(self, k):
                kwarg |= {k: getattr(self, k)}

        self._calculate = __curry__(func)(**kwarg)


def kmeans(points: ndarray, n_clusters: int, random_state=0, **kwargs):
    """
    p_function:  this function with parameter, supports layer-by-layer analytics
    :param f:  some function, in case: a nx2 matrix of coordinates
    :param p: p - parameter, in function - clusters count
    :return: kmeans (sklearn.cluster.KMeans) , you can get :
    kmeans.labels_,
    kmeans.cluster_centers_
    more info in the scikit-learn documentation
    """

    return KMeans(n_clusters, random_state=random_state, **kwargs).fit(points)
