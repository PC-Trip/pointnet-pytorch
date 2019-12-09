# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import pymesh
from tqdm import tqdm
from utils import shapenet_labels
import logging
from time import time
import open3d as o3d


def scale_linear_bycolumn(rawdata, high=1.0, low=0.0):
    mins = np.min(rawdata, axis=0)
    maxs = np.max(rawdata, axis=0)
    rng = maxs - mins
    return high - (high-low)*(maxs-rawdata)/(rng+np.finfo(np.float32).eps)


class ClsDataset(data.Dataset):
        '''Object classification on ModelNet'''
        def __init__(self, root, npoints=1024, train=True):
                self.root = root
                self.npoints = npoints
                self.catfile = os.path.join(self.root, 'modelnet_cat2num.txt')
                self.cat = {}

                with open(self.catfile, 'r') as f:
                        for line in f.readlines():
                                lns = line.strip().split()
                                self.cat[lns[0]] = lns[1]
                self.num_classes = len(self.cat)
                self.datapath = []
                FLAG = 'train' if train else 'test'
                for item in os.listdir(self.root):
                        if os.path.isdir(os.path.join(self.root, item)):
                                for f in os.listdir(os.path.join(self.root, item, FLAG)):
                                    if f.endswith('.off'):
                                        self.datapath.append((os.path.join(self.root, item, FLAG, f), int(self.cat[item])))


        def __getitem__(self, idx):
                fn = self.datapath[idx]
                points = pymesh.load_mesh(fn[0]).vertices
                label = fn[1]
                replace = True if points.shape[0]<self.npoints else False
                choice = np.random.choice(points.shape[0], self.npoints, replace=replace)
                points = points[choice, :]
                points = scale_linear_bycolumn(points)
                points = torch.from_numpy(points.astype(np.float32))
                label = torch.from_numpy(np.array([label]).astype(np.int64))
                return points, label


        def __len__(self):
                return len(self.datapath)


class PartDataset(data.Dataset):
        def __init__(self, root, npoints=2048, class_choice=None, train=True):
                '''Part segmentation on ShapeNet'''
                self.root = root
                self.npoints = npoints
                self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
                self.cat = {}

                with open(self.catfile, 'r') as f:
                        for line in f.readlines():
                                lns = line.strip().split()
                                self.cat[lns[0]] = lns[1]
                
                if not class_choice is None:
                    self.cat = {k:v for k, v in self.cat.items() if k in class_choice}
                    self.num_classes = shapenet_labels[class_choice[0]]
                else:
                    self.num_classes = 50

                self.meta = {}
                for item in self.cat:
                        self.meta[item] = []
                        point_dir = os.path.join(self.root, self.cat[item], 'points')
                        seg_dir = os.path.join(self.root, self.cat[item], 'points_label')
                fns = sorted(os.listdir(point_dir))
                if train:
                        fns = fns[:int(0.9*len(fns))]
                else:
                        fns = fns[int(0.9*len(fns)):]

                for fn in fns:
                        token = (os.path.splitext(os.path.basename(fn))[0])
                        self.meta[item].append((os.path.join(point_dir, token + '.pts'), os.path.join(seg_dir, token + '.seg')))

                self.datapath = []
                for item in self.cat:
                        for fn in self.meta[item]:
                                self.datapath.append((item, fn[0], fn[1]))

                self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))


        def __getitem__(self, idx):
                fn = self.datapath[idx]
                points = np.loadtxt(fn[1]).astype(np.float32)
                seg = np.loadtxt(fn[2]).astype(np.int64)
                replace = True if points.shape[0]<self.npoints else False
                choice = np.random.choice(len(seg), self.npoints, replace=replace)
                # resample
                points = points[choice, :]
                seg = seg[choice]
                points = torch.from_numpy(points)
                seg = torch.from_numpy(seg)
                return points, seg


        def __len__(self):
                return len(self.datapath)


class S3dDataset(data.Dataset):
        '''Semantic segmentation on S3DIS'''
        def __init__(self, root, npoints=4096, train=True, gen_labels=False):
                self.root = root
                self.npoints = npoints
                self.catfile = os.path.join(self.root, 's3d_cat2num.txt')
                self.cat = {}
                with open(self.catfile, 'r') as f:
                    for line in f.readlines():
                        lns = line.strip().split()
                        self.cat[lns[0]] = lns[1]
                self.num_classes = len(self.cat)
                self.datapath, self.labelspath = [], []
                FLAG = 'train' if train else 'test'
                path = os.path.join(self.root, FLAG)
                for area in os.listdir(path):
                    area_path = os.path.join(path, area)
                    for scene in os.listdir(area_path):
                        if os.path.isdir(os.path.join(area_path, scene)):
                            scene_path = os.path.join(area_path, scene)
                            for scene_component in os.listdir(os.path.join(scene_path, 'Annotations')):
                                if 'labels' not in scene_component and 'preds' not in scene_component:
                                    self.datapath.append(os.path.join(scene_path, 'Annotations', scene_component))
                

                if gen_labels: # do this only once
                    pbar = tqdm(total=len(self.datapath))
                    ds_i = 0
                    for path in self.datapath:
                        l = path.split('\\')
                        labels_path = os.path.join(l[-5], l[-4], l[-3], l[-2])
                        component_name = l[-1].split('.')[0]
                        class_name = l[-1].split('_')[0]
                        if class_name == '.DS':
                            ds_i += 1
                            continue
                        file_path = os.path.join(labels_path, component_name + '_labels.txt')
                        if os.path.exists(file_path):
                            continue
                        with open(path, 'r') as f:
                            with open(file_path, 'w') as g:
                                for line in f.readlines():
                                    g.write(str(self.cat[class_name]) + '\n')
                        pbar.update()
                    

        def __getitem__(self, idx):
            try:
                fn = self.datapath[idx]
                points = np.loadtxt(fn)[:, :3].astype(np.float32)
                ln = os.path.splitext(fn)[0] + '_labels.txt'
                seg = np.loadtxt(ln).astype(np.int64)
                replace = True if points.shape[0]<self.npoints else False
                choice = np.random.choice(points.shape[0], self.npoints, replace=replace)
                points = points[choice, :]
                points = scale_linear_bycolumn(points)
                seg = seg[choice]
                points = torch.from_numpy(points)
                seg = torch.from_numpy(seg)
                return points, seg
            
            except Exception as e:
                    print(fn)
                    print(e)
                    raise e


        def __len__(self):
            return len(self.datapath)


class RoomsDataset_mk1:
        '''
                Semantic segmentation on S3DIS

                take sample from a room only ones (simple)
        '''

        def __init__(self, path, npoints=4096, feature_num=3):
                self.path = path
                self.files_paths = [os.path.join(self.path, fn) for fn in os.listdir(path)]
                self.npoints = npoints
                self.feature_num = feature_num

        def __len__(self):
                return len(self.files)

        def __getitem__(self, idx):
                fp = self.files_paths[idx]
                points = np.loadtxt(fp)[:, :3]
                seg = points[:, -1].astype(np.int64)
                points = points[:, :feature_num].astype(np.float32)
                replace = True if points.shape[0]<self.npoints else False
                choice = np.random.choice(points.shape[0], self.npoints, replace=replace)
                points = points[choice, :]
                points = scale_linear_bycolumn(points)
                seg = seg[choice]
                points = torch.from_numpy(points)
                seg = torch.from_numpy(seg)
                return points, seg




class RoomsDataset_mk2:
        '''
                Semantic segmentation on S3DIS

                types of room slicing:
                1) x_size*y_size*z_size
                2) x_mesh_num*y_mesh_num*z_mesh_num


                we have to return a len. we don't want to load all Gbs of data to calculate size
                let me set some size and when we are out of it we cycle (idx % real_size) until
                we met that size. 
        '''

        def __init__(self, path, npoints=4096, size=1000, feature_num=3, slicing_sizes=(1.0, 1.0, 1.0), slicing_mesh=None):
                self.path = path
                self.files_paths = [os.path.join(self.path, fn) for fn in os.listdir(self.path)]
                random.shuffle(self.files_paths)

                self.npoints = npoints
                self.size = size
                # self.feature_num = feature_num
                self.box_gen = None
                self.slicing_sizes = slicing_sizes
                self.slicing_mesh = slicing_mesh
                if slicing_sizes and slicing_mesh:
                        raise Exception("Set only one type of room slicing, please")


        def __len__(self):
                return self.size


        def make_box_gen(self, points):
                """
                Split points in to subarrays to list

                input: points - np.array of shape (point_number, xyz[rgb]+label)
                """
                boxes = []
                if self.slicing_sizes:
                        # there are 8 points to start sized slicing... we will go from the mins to max
                        # also we can start from random point of view
                        mins = np.min(points[:, :3], axis=0)
                        maxs = np.max(points[:, :3], axis=0)
                        ranges = maxs-mins
                        ns=(ranges/np.array(self.slicing_sizes)).astype(np.int64)
                        ns = np.clip(ns, 0, None)
                        for ix in range(ns[0]):
                                for iy in range(ns[1]):
                                        for iz in range(ns[2]):
                                                # print(ix, iy, iz)
                                                # print(self.slicing_sizes)
                                                # print(ns)
                                                # print(ranges)
                                                x_indices = (mins[0]+self.slicing_sizes[0]*ix<points[:, 0]) * (mins[0]+self.slicing_sizes[0]*(ix+1)>points[:, 0])
                                                y_indices = (mins[1]+self.slicing_sizes[1]*iy<points[:, 1]) * (mins[1]+self.slicing_sizes[1]*(iy+1)>points[:, 1])
                                                z_indices = (mins[2]+self.slicing_sizes[2]*iz<points[:, 2]) * (mins[2]+self.slicing_sizes[2]*(iz+1)>points[:, 2])
                                                # print(np.sum(x_indices), np.sum(y_indices), np.sum(z_indices))
                                                # print(np.sum(x_indices*y_indices))
                                                # print(np.sum(x_indices*z_indices))
                                                # print(np.sum(y_indices*z_indices))
                                                # print(np.sum(x_indices*y_indices*z_indices))
                                                indices = x_indices*y_indices*z_indices
                                                # print('indshp', indices.shape)
                                                if np.sum(indices) == 0:
                                                        continue
                                                yield points[indices, :]


        def new_box_gen(self):
                if len(self.files_paths) == 0:
                        self.files_paths =[os.path.join(self.path, fn) for fn in os.listdir(self.path)]
                        random.shuffle(self.files_paths)
                fp = self.files_paths.pop(0)
                print("Loading file: {}".format(fp))
                points = np.loadtxt(fp)
                self.box_gen = self.make_box_gen(points)


        def reset(self):
                self.files_paths = []
                self.new_box_gen()


        def __getitem__(self, idx):
                if self.box_gen is None:
                        self.new_box_gen()
                
                try:
                        box = next(self.box_gen)
                except StopIteration:
                        self.new_box_gen()
                        box = next(self.box_gen)

                points = box
                if self.npoints is not None:
                        replace = True if box.shape[0]<self.npoints else False
                        choice = np.random.choice(box.shape[0], self.npoints, replace=replace)
                        points = box[choice, :]

                xyz = points[:, :6].astype(np.float32)
                seg = points[:, -1].astype(np.int64)
                xyz = scale_linear_bycolumn(xyz)
                return xyz, seg


class S3dDatasetNeiSphe(data.Dataset):
    def __init__(self, root, npoints=4096, train=True, gen_labels=False):
        self.root = root
        self.npoints = npoints
        self.catfile = os.path.join(self.root, 's3d_cat2num.txt')
        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f.readlines():
                lns = line.strip().split()
                self.cat[lns[0]] = lns[1]
        self.num_classes = len(self.cat)
        self.datapath, self.labelspath = [], []
        FLAG = 'train' if train else 'test'
        path = os.path.join(self.root, FLAG)
        for area in os.listdir(path):
            area_path = os.path.join(path, area)
            for scene in os.listdir(area_path):
                if os.path.isdir(os.path.join(area_path, scene)):
                    scene_path = os.path.join(area_path, scene)
                    for scene_component in os.listdir(
                            os.path.join(scene_path, 'Annotations')):
                        if not scene_component.endswith(
                                '_labels.txt') and not scene_component.startswith(
                                '.'):
                            self.datapath.append(
                                os.path.join(scene_path, 'Annotations',
                                             scene_component))

        if gen_labels:  # do this only once
            for path in tqdm(self.datapath):
                dir_name, base_name = os.path.split(path)
                root, ext = os.path.splitext(base_name)
                class_name = root.split('_')[0]
                file_path = os.path.join(dir_name, root + '_labels.txt')
                with open(os.path.join(file_path), 'w') as f:
                    f.write(str(self.cat[class_name]))

    @pc_normalize(norm_type='spherical', center=False, verbose=False)
    @ps_to_spherical(verbose=False)
    # @pc_info(cmap='tab10', viz=True)
    @pc_noise(sigma=0.01, seed=42)
    @pc_rotate(max_x=0, max_y=0, max_z=180, seed=42)
    def __getitem__(self, idx):
        n = 0
        while n != self.npoints:
            fn = self.datapath[idx]
            points = np.loadtxt(fn)[:, :3].astype(np.float32)
            ln = os.path.splitext(fn)[0] + '_labels.txt'
            seg = np.loadtxt(ln).astype(np.int64)
            seg = np.full(points.shape[0], seg)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            idx = np.random.randint(0, points.shape[0])
            n, idxs, ds = pcd_tree.search_knn_vector_3d(points[idx], self.npoints)
            idx = np.random.randint(0, len(self.datapath))  # FIXME workaround
        points = points[idxs]
        seg = seg[idxs]
        points = torch.from_numpy(points)
        seg = torch.from_numpy(seg)
        return points, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':

        # c = ClsDataset(root='modelnet40_manually_aligned')
        # ps, seg = c[100]
        # print(ps.type(), ps.size(), l.type(), l.size(), l)

        # d = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0')
        # ps, seg = d[10]
        # print(ps.type(), ps.size(), seg.type(), seg.size())

        s = S3dDataset(root='.', train=True, gen_labels=True)
        s = S3dDataset(root='.', train=False, gen_labels=True)
        s = S3dDatasetNeiSphe(root='Stanford3dDataset_v1.2', train=True, gen_labels=True)
        ps, seg = s[100]
        print(ps.type(), ps.size(), seg.type(), seg.size())
        for _ in tqdm(s):
            try:
                pass
            except Exception as e:
                print(e)


