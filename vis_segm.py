import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d

def vis_all():
    files_path   = 'D:\\code\\pytorch\\pointnet-pytorch\\train\\Area_1\\conferenceRoom_2\\Annotations'
    n_classes = 14
    encountered_classes = set()


    all_files    = os.listdir(files_path)
    points_files = list(filter(lambda x: 'labels' not in x and 'preds' not in x, all_files))
    labels_files = list(filter(lambda x: 'labels' in x, all_files))
    predics_files = list(filter(lambda x: 'preds' in x, all_files))
    cmap = cm.get_cmap('jet')

    pcds = []
    for points_file in points_files:
        obj_name = points_file.split('.')[0]
        preds_file = [f for f in predics_files if obj_name in f][0]
        points = np.loadtxt(os.path.join(files_path, points_file)).astype(np.float32)
        preds = np.loadtxt(os.path.join(files_path, preds_file)).astype(np.float32)

        colors = np.zeros((points.shape[0], 3))
        for i, c in enumerate(preds):
            encountered_classes.add(c)
            if i != -1:  # noise (black color by default)
                color = cmap(int(c) / n_classes)[:-1]  # remove alpha
                colors[i] = color

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = open3d.utility.Vector3dVector(colors)
        pcds.append(pcd)

    print(encountered_classes)
    open3d.visualization.draw_geometries(pcds)


def vis_one():
    points_file = 'D:\\code\\pytorch\pointnet-pytorch\\train\Area_2\\auditorium_1\\Annotations\\board_1.txt'
    preds_file   = 'D:\\code\\pytorch\pointnet-pytorch\\train\Area_2\\auditorium_1\\Annotations\\board_1_preds.txt'
    n_classes = 14

    cmap = cm.get_cmap('jet')

    pcds = []
    points = np.loadtxt(points_file).astype(np.float32)
    preds = np.loadtxt(preds_file).astype(np.float32)

    colors = np.zeros((points.shape[0], 3))
    for i, c in enumerate(preds):
        if i != -1:  # noise (black color by default)
            color = cmap(int(c) / n_classes)[:-1]  # remove alpha
            colors[i] = color

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = open3d.utility.Vector3dVector(colors)
    pcds.append(pcd)

    open3d.visualization.draw_geometries(pcds)

if __name__ == "__main__":
    vis_one()