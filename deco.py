import open3d as o3d
from matplotlib import cm
import numpy as np


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


def pc_info(cmap='jet', viz=True):
    cmap = cm.get_cmap(cmap)

    def deco(func):
        def wrapper(*args, **kwargs):
            # Preprocessing
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
            print('max: {}'.format(points.max(0)))
            print('min: {}'.format(points.min(0)))
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
                        colors[i] = cmap(c / (n_classes - 1))[
                                    :-1]  # remove alpha
                else:
                    colors = np.full((seg.shape[0], 3), cmap(0)[:-1])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis = o3d.visualization.VisualizerWithEditing()
                c = Callback(seg)
                vis.register_animation_callback(c)
                vis.create_window(width=500, height=500, left=800, top=50)
                vis.add_geometry(pcd)
                vis.run()
                vis.destroy_window()
            return points, seg

        return wrapper

    return deco
