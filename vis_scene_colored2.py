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


def vis_numpy(points):
    classes = np.unique(points[:, -1])
    cmap = cm.get_cmap('jet')

    pcds = []
    for i, class_num in enumerate(classes):
        classed_points = points[points[:, -1]==class_num]
        colors = np.zeros((classed_points.shape[0], 3))
        colors[:] = cmap(i / len(classes))[:-1]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(classed_points[:, :3])
        pcd.colors = open3d.utility.Vector3dVector(colors)
        pcds.append(pcd)

    open3d.visualization.draw_geometries(pcds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '--path', type=str, required=True, help='path to points file')
    points = np.loadtxt(parser.parse_args().path)
    vis_numpy(points)
