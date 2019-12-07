import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from datasets import S3dDataset
from pointnet import PointNetSeg


import matplotlib.pyplot as plt
from matplotlib import cm
import open3d

from utils import get_s3d_cat2num




if __name__ == '__main__':
    room_path = "D:\\code\\pytorch\\pointnet-pytorch\\train\Area_1\\conferenceRoom_2\\Annotations"
    files = os.listdir(room_path)
    points_files = list(filter(lambda x: 'label' not in x and 'pred' not in x, files))
    cat2num = get_s3d_cat2num()
    n_classes = len(cat2num)
    cmap = cm.get_cmap('jet')

    pcds = []
    for points_file in points_files:
        cat = cat2num[points_file.split('_')[0]]
        points = np.loadtxt(os.path.join(room_path, points_file)).astype(np.float32)

        colors = np.zeros((points.shape[0], 3))
        colors[:] = cmap(int(cat) / n_classes)[:-1]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = open3d.utility.Vector3dVector(colors)
        pcds.append(pcd)

    open3d.visualization.draw_geometries(pcds)