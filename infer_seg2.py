# -*- coding: utf-8 -*-

import argparse
import os
import pdb

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

def scale_linear_bycolumn(rawdata, high=1.0, low=0.0):
    mins = np.min(rawdata, axis=0)
    maxs = np.max(rawdata, axis=0)
    rng = maxs - mins
    return high - (high-low)*(maxs-rawdata)/(rng+np.finfo(np.float32).eps)



def infer_one_file(config):

    blue = lambda x: '\033[94m' + x + '\033[0m'
    yellow = lambda x: '\033[93m' + x + '\033[0m'
    red = lambda x: '\033[91m' + x + '\033[0m'

    classifier = PointNetSeg(k=config['num_classes'])
    classifier.load_state_dict(torch.load(config['model']))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    classifier.to(device)

    points = np.loadtxt(config['object'])[:, :3].astype(np.float32)
    choice = np.random.choice(points.shape[0], config['npoints'], replace=True)
    pred = np.zeros((points.shape[0],))
    pred[:] = -1 
    points = points[choice, :]
    points = np.expand_dims(points, axis=0)
    points = torch.from_numpy(points)
    points = points.transpose(2, 1)
    points = points.to(device)
    classifier = classifier.eval()
    choiced_pred, _ = classifier(points)
    choiced_pred = choiced_pred.view(-1, config['num_classes'])
    choiced_pred = choiced_pred.data.max(1)[1]

    cat2num = get_s3d_cat2num()
    for el in set(choiced_pred.tolist()):
        print("found {}".format(cat2num[el]))

    pred[choice] = choiced_pred 
    np.savetxt(config['outf'], pred)


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


def vis_one(points_file, preds_file):
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '--path', type=str, required=True, help='path to points file')
    parser.add_argument('-model', '--model', type=str, default=None, help='path to model file')
    config = parser.parse_args()
    infer(config)

