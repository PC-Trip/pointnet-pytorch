import os
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
# import open3d as o3d
from deco2 import pc_normalize
import pickle


class S3dDataset(data.Dataset):
    '''Semantic segmentation on S3DIS'''

    def __init__(self, root, npoints=4096, train=True, gen_labels=False,
                 memoize=True, load=True):
        self.root = root
        self.npoints = npoints
        self.catfile = os.path.join(self.root, 's3d_cat2num.txt')
        self.cat = {x.split()[0]: int(x.split()[1]) for x in open(self.catfile)}
        self.num_classes = len(self.cat)
        self.datapath, self.labelspath = [], []
        self.memoize = memoize
        self.ds_type = 'train' if train else 'test'
        path = os.path.join(self.root, self.ds_type)
        # cnt = 0
        for root, dirs, _ in os.walk(path):
            for d in dirs:
                if d == 'Annotations':
                    # cnt += 1
                    # if cnt > 3:
                    #     break
                    components = os.listdir(os.path.join(root, d))
                    for c in components:
                        name, ext = os.path.splitext(c)
                        if not name.startswith('.') and not name.endswith(
                                'labels'):
                            self.datapath.append(os.path.join(root, d, c))
        print('components: {}'.format(len(self.datapath)))
        if self.memoize:
            if load:
                print('Loading memoization')
                with open(os.path.join(self.root, 'mem_points_{}.pkl'.format(
                        self.ds_type)), 'rb') as f:
                    self.points = pickle.load(f)
                with open(os.path.join(self.root, 'mem_labels_{}.pkl'.format(
                        self.ds_type)), 'rb') as f:
                    self.labels = pickle.load(f)
            else:
                print('Memoizing')
                self.points = {}
                self.labels = {}
                for i, dp in enumerate(self.datapath):
                    print('{}/{} {}'.format(i + 1, len(self.datapath), dp))
                    d = np.loadtxt(dp)
                    points = d[:, :3].astype(np.float32)
                    label = os.path.splitext(os.path.basename(dp))[0].split('_')[0]
                    self.labels[dp] = self.cat[label]
                    self.points[dp] = points
                print('Saving memoization')
                with open(os.path.join(self.root, 'mem_points_{}.pkl'.format(
                        self.ds_type)), 'wb') as f:
                    pickle.dump(self.points, f)
                with open(os.path.join(self.root, 'mem_labels_{}.pkl'.format(
                    self.ds_type)), 'wb') as f:
                    pickle.dump(self.labels, f)
        else:
            pass
            # if gen_labels:  # do this only once
            #     pbar = tqdm(total=len(self.datapath))
            #     ds_i = 0
            #     for path in self.datapath:
            #         l = path.split('\\')
            #         labels_path = os.path.join(l[-5], l[-4], l[-3], l[-2])
            #         component_name = l[-1].split('.')[0]
            #         class_name = l[-1].split('_')[0]
            #         if class_name == '.DS':
            #             ds_i += 1
            #             continue
            #         file_path = os.path.join(labels_path,
            #                                  component_name + '_labels.txt')
            #         if os.path.exists(file_path):
            #             continue
            #         with open(path, 'r') as f:
            #             with open(file_path, 'w') as g:
            #                 for line in f.readlines():
            #                     g.write(str(self.cat[class_name]) + '\n')
            #         pbar.update()

    @pc_normalize(verbose=False)
    def __getitem__(self, idx):
        fn = self.datapath[idx]
        # print(fn)
        if self.memoize:
            points = self.points[fn]
            labels = self.labels[fn]
            replace = True if points.shape[0] < self.npoints else False
            choice = np.random.choice(points.shape[0], self.npoints,
                                      replace=replace)
            points = points[choice, :]
            if isinstance(labels, int):
                seg = np.full(points.shape[0], labels)
            else:
                seg = labels[choice]
            points = torch.from_numpy(points)
            seg = torch.from_numpy(seg)
            return points, seg
        else:
            pass
            # points = np.loadtxt(fn)[:, :3].astype(np.float32)
            # ln = os.path.splitext(fn)[0] + '_labels.txt'
            # seg = np.loadtxt(ln).astype(np.int64)
            # replace = True if points.shape[0] < self.npoints else False
            # choice = np.random.choice(points.shape[0], self.npoints,
            #                           replace=replace)
            # points = points[choice, :]
            # seg = seg[choice]
            # points = torch.from_numpy(points)
            # seg = torch.from_numpy(seg)
            # return points, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    train_ds = S3dDataset(root='Stanford3dDataset_v1.2', train=True, gen_labels=False,
                   memoize=True, load=True)
    test_ds = S3dDataset(root='Stanford3dDataset_v1.2', train=False, gen_labels=False,
                   memoize=True, load=True)
    ps, seg = train_ds[-1]
    print(ps.type(), ps.size(), seg.type(), seg.size())
    ps, seg = test_ds[-1]
    print(ps.type(), ps.size(), seg.type(), seg.size())
    # for _ in tqdm(s):
    #     try:
    #         pass
    #     except Exception as e:
    #         print(e)
