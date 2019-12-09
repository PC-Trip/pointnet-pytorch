import open3d as o3d
import numpy as np
from matplotlib import cm


class Callback:
    def __init__(self, point_to_class_map, classes_names):
        self.points = set()
        self.ptc_map = point_to_class_map
        self.cs_ns = classes_names

    def __call__(self, vis):
        idxs = vis.get_picked_points()
        for i in idxs:
            if i not in self.points:
                self.points.add(i)
                pc = self.ptc_map[i]
                pn = self.cs_ns[pc]
                print('{} point class: {} - {}'.format(i, pc, pn))
        return False


if __name__ == '__main__':
    file_name = 's3d2/test/rooms/WC_2.txt'
    print(file_name)
    data = np.loadtxt(file_name)
    labels_names = [x.split()[0] for x in open('s3d_cat2num.txt')]
    print(labels_names)
    points, colors, labels = data[:, :3], data[:, 3:6] / 255, data[:, 6].astype(
        np.int)
    print(points)
    print(colors)
    print(labels)
    print('n_points: {}'.format(points.shape[0]))
    unique_labels = np.unique(labels)
    print('labels: {}'.format(unique_labels))
    print('labels_names: {}'.format([labels_names[x] for x in unique_labels]))
    labels_cnt = np.bincount(labels)
    labels_cnt = labels_cnt[labels_cnt != 0]
    print('\n'.join('{}: {}'.format(labels_names[y], x) for x, y in
                    zip(labels_cnt, unique_labels)))
    n_labels = len(unique_labels)
    max_label = max(unique_labels)
    print('n_labels: {}'.format(n_labels))
    cmap = cm.get_cmap('tab10')
    if n_labels > 1:
        labels_colors = np.apply_along_axis(
            lambda x: cmap(x / max_label), 0, labels)[:, :3]
    else:
        labels_colors = np.full_like(labels, cmap(0)[:-1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(labels_colors)
    vis = o3d.visualization.VisualizerWithEditing()
    c = Callback(labels, labels_names)
    vis.register_animation_callback(c)
    vis.create_window(width=500, height=500,
                      left=50, top=50)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()
