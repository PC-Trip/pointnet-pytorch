import open3d as o3d
from matplotlib import cm
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class Callback:
    def __init__(self, point_to_class_map):
        self.points = set()
        self.ptc_map = point_to_class_map

    def __call__(self, vis):
        idxs = vis.get_picked_points()
        for i in idxs:
            if i not in self.points:
                self.points.add(i)
                print('{} point class: {}'.format(i, self.ptc_map[i]))
        return False


def pc_info(cmap='tab10', viz=True, coord_axes=True, black_bg=True,
            width=500, height=500, left=800, top=50):
    cmap = cm.get_cmap(cmap)

    def deco(func):
        def wrapper(*args, **kwargs):
            # Preprocessing
            print('pc_info')
            dataset, idx = args
            fn = dataset.datapath[idx]
            print(fn)
            # FIXME can't add multiple point clouds to the VisualizerWithEditing
            # all_points = np.loadtxt(fn)[:, :3].astype(np.float32)
            # points_min = np.min(all_points, axis=0)
            # points_ptp = np.ptp(all_points, axis=0)  # max - min
            # print(points_min, points_ptp)
            # # if norm_type == 'coord_max':
            # all_points = (all_points - points_min) / points_ptp
            # all_colors = np.zeros((all_points.shape[0], 3))
            # # else:
            # #     pcd.points = o3d.utility.Vector3dVector(
            # #         (pcd.points - points_min) / max(points_ptp))
            # all_pcd = o3d.geometry.PointCloud()
            # all_pcd.points = o3d.utility.Vector3dVector(all_points)
            # all_pcd.colors = o3d.utility.Vector3dVector(all_colors)
            points, seg = func(*args, **kwargs)
            classes = seg.unique()
            n_classes = len(classes)
            # Show info
            print('min: {}'.format(points.min(0)))
            print('max: {}'.format(points.max(0)))
            print('n_points: {}'.format(points.shape[0]))
            print('classes: {}'.format(classes))
            print('n_classes: {}'.format(n_classes))
            if viz:  # Show cloud
                # Colorize
                # FIXME doesn't work
                # colors = np.apply_along_axis(
                #     lambda x: cmap(x / (n_classes - 1))[:-1]
                #     if n_classes > 1 else cmap(0)[:-1], 0, seg.numpy())
                if n_classes > 1:
                    colors = np.zeros((seg.shape[0], 3))
                    for i, c in enumerate(seg):
                        colors[i] = cmap(c / (n_classes - 1))[:-1]  # -alpha
                else:
                    colors = np.full((seg.shape[0], 3), cmap(0)[:-1])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis = o3d.visualization.VisualizerWithEditing()
                c = Callback(seg)
                vis.register_animation_callback(c)
                vis.create_window(width=width, height=height,
                                  left=left, top=top)
                vis.add_geometry(pcd)
                opt = vis.get_render_option()
                if coord_axes:
                    # FIXME move frame to 0, 0, 0
                    opt.show_coordinate_frame = True
                if black_bg:
                    opt.background_color = np.asarray([0, 0, 0])
                vis.run()
                vis.destroy_window()
            return points, seg

        return wrapper

    return deco


def pc_normalize(norm_type='global', center=True, verbose=False):
    """
    Point normalization decorator for Dataset.__getitem__
    :param str norm_type:
    global: max_range = max(max(X)-min(X), max(Y)-min(Y), max(Z)-min(Z))
        X-min(X)/max_range, Y-min(Y)/max_range, Z-min(Z)/max_range
    local: X-min(X)/max(X)-min(X), Y-min(Y)/max(Y)-min(Y), Z-min(Z)/max(Z)-min(Z)
    spherical: R-min(R)/range(R), THETA/PI, PHI-(-PI)/2*PI
    :param bool center: center around 0, 0, 0?
    :param bool verbose:
    :return: points, seg
    """

    def deco(func):
        def wrapper(*args, **kwargs):
            points, seg = func(*args, **kwargs)
            points_min = points.min(0)[0]
            points_max = points.max(0)[0]
            points_range = points_max - points_min
            if verbose:
                print('pc_normalize')
                print('min: {}'.format(points_min))
                print('max: {}'.format(points_max))
                print('range: {}'.format(points_range))
            if norm_type == 'global':
                max_range = points_range.max()
                if max_range != 0:
                    points = (points - points_min) / max_range
            elif norm_type == 'local':
                points_range[points_range == 0] = 1  # replace 0 to 1
                points = (points - points_min) / points_range
            elif norm_type == 'spherical':
                points_range[points_range == 0] = 1
                points_range[1] = np.pi
                points_range[2] = 2 * np.pi
                points_min[1] = 0.0
                points_min[2] = -np.pi
                points = (points - points_min) / points_range
            else:
                raise ValueError('Wrong normalization type: {},'
                                 ' choose global or local'.format(norm_type))
            if center:
                points_min = points.min(0)[0]
                points_max = points.max(0)[0]
                points_range = points_max - points_min
                points = 2 * points - points_range
            if verbose:
                points_min = points.min(0)[0]
                points_max = points.max(0)[0]
                points_range = points_max - points_min
                print('new_min: {}'.format(points_min))
                print('new_max: {}'.format(points_max))
                print('new_range: {}'.format(points_range))
            return points, seg

        return wrapper

    return deco


def pc_noise(sigma=0.01, seed=None):
    def deco(func):
        def wrapper(*args, **kwargs):
            points, seg = func(*args, **kwargs)
            if seed is not None:
                torch.manual_seed(seed)
            points = torch.distributions.normal.Normal(
                points, torch.full_like(points, sigma)).sample()
            return points, seg

        return wrapper

    return deco


def pc_rotate(max_x=30, max_y=30, max_z=30, seed=None):
    def deco(func):
        def wrapper(*args, **kwargs):
            points, seg = func(*args, **kwargs)
            # r = torch.from_numpy(R.random().as_dcm().astype(np.float32))
            if seed is not None:
                np.random.seed(seed)
            ax = np.random.uniform(-max_x, max_x)
            ay = np.random.uniform(-max_y, max_y)
            az = np.random.uniform(-max_z, max_z)
            r = torch.from_numpy(
                R.from_euler('xyz', [ax, ay, az],
                             degrees=True).as_dcm().astype(np.float32))
            # print(r)
            points = torch.mm(points, r)
            return points, seg

        return wrapper

    return deco


def ps_to_spherical(center=None, verbose=True):
    def deco(func):
        def wrapper(*args, **kwargs):
            points, seg = func(*args, **kwargs)
            if center is None:
                c = points[0].numpy().copy()
            else:
                c = center
            for i, p in enumerate(points):  # TODO optimize
                x, y, z = p.numpy() - c
                r = np.sqrt(x * x + y * y + z * z)
                theta = np.arccos(z / r) if r != 0 else 0.0
                phi = np.arctan2(y, x)
                points[i] = torch.from_numpy(np.array([r, theta, phi]))
            if verbose:
                points_min = points.min(0)[0]
                points_max = points.max(0)[0]
                points_range = points_max - points_min
                print('ps_to_spherical')
                print('min: {}'.format(points_min))
                print('max: {}'.format(points_max))
                print('range: {}'.format(points_range))
            return points, seg

        return wrapper

    return deco
