import open3d as o3d
import numpy as np
import torchvision
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


def read_cloud(file_name):
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
    return points, labels, labels_colors, labels_names


def plot_all(file_name):
    points, labels, labels_colors, labels_names = read_cloud(file_name)
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


def plot_2(file_name):
    points, labels, labels_colors, labels_names = read_cloud(file_name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    n, idxs, ds = pcd_tree.search_hybrid_vector_3d(points[0], 2, 100000)
    print(n, len(idxs), len(ds))
    ps2 = points[idxs]
    ls2 = labels[idxs]
    cs2 = labels_colors[idxs]
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(ps2)
    pcd2.colors = o3d.utility.Vector3dVector(cs2)
    vis = o3d.visualization.VisualizerWithEditing()
    c = Callback(ls2, labels_names)
    vis.register_animation_callback(c)
    vis.create_window(width=500, height=500, left=50, top=50)
    vis.add_geometry(pcd2)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 3000)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.fc4(x))
        return x


import os


def train_test():
    torch.manual_seed(0)
    net = Net()
    print(net)
    if os.path.exists('net.pth'):
        net.load_state_dict(torch.load('net.pth'))
        net.eval()
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        x = torch.randn(3, 3)
        t = torch.randn(3, 3)
        for i in range(100):
            optimizer.zero_grad()
            p = net(x)
            loss = criterion(p, t)
            loss.backward()
            optimizer.step()
            print(loss)
    # params = list(net.parameters())
    for param in net.parameters():
        print(type(param.data), param.size())
    torch.save(net.state_dict(), 'net.pth')
    # x2
    # with torch.no_grad():
    x2 = net.fc3(torch.randn(3, 256))
    x2 = net.fc4(x2)
    print(x2)


import torch.utils.data as data


class Dataset2(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.points, self.labels, self.labels_colors, self.labels_names = read_cloud(root)
        self.transform = transform
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    def __getitem__(self, idx):
        # print(idx)
        max_nn = 100000000
        # idx = np.random.randint(len(self.points))
        n, idxs, ds = self.pcd_tree.search_hybrid_vector_3d(
            self.points[idx], 0.1, max_nn)
        # print(idxs)
        # print(len(idxs))
        # print(ds)
        if len(idxs) >= 1000:
            filtered_idxs = np.random.choice(len(idxs), 1000, replace=False)
        else:
            filtered_idxs = np.random.choice(len(idxs), 1000, replace=True)
        filtered_idxs.sort()
        # print(filtered_idxs)
        idxs = [idxs[x] for x in filtered_idxs]
        # print(idxs)
        # print(n, len(idxs), len(ds))
        ps2 = self.points[idxs]
        ls2 = self.labels[idxs]
        sample = {'points': ps2, 'labels': ls2}
        # cs2 = self.labels_colors[idxs]
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(sample['points'])
        # pcd2.colors = o3d.utility.Vector3dVector(cs2)
        # vis = o3d.visualization.VisualizerWithEditing()
        # c = Callback(sample['labels'], self.labels_names)
        # vis.register_animation_callback(c)
        # vis.create_window(width=500, height=500, left=50, top=50)
        # vis.add_geometry(pcd2)
        # opt = vis.get_render_option()
        # opt.show_coordinate_frame = True
        # opt.background_color = np.asarray([0, 0, 0])
        # vis.run()
        # vis.destroy_window()
        if self.transform:
            sample = self.transform(sample)
        # print(sample['points'])
        return sample

    def __len__(self):
        return len(self.points)


class Normalize(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        ps, ls = sample['points'], sample['labels']
        # print('Norm')
        sample = {'points': ps, 'labels': ls}
        return sample


class ToSpherical(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        ps, ls = sample['points'], sample['labels']
        # print('ToSpherical')
        c = ps[0].copy()
        for i, p in enumerate(ps):  # TODO optimize
            x, y, z = p - c
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r) if r != 0 else 0.0
            phi = np.arctan2(y, x)
            ps[i] = np.array([r, theta, phi])
        ps_min = ps.min(0)
        ps_max = ps.max(0)
        ps_range = ps_max - ps_min
        ps_range[ps_range == 0] = 1
        ps_range[1] = np.pi
        ps_range[2] = 2 * np.pi
        ps_min[1] = 0.0
        ps_min[2] = -np.pi
        ps = (ps - ps_min) / ps_range
        sample = {'points': ps, 'labels': ls}
        return sample

from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler, RandomSampler

def train(dataset):
    torch.manual_seed(0)
    net = Net()
    print(net)
    # for param in net.parameters():
    #     print(param)
    batch_size = 30
    n_epochs = 10
    if os.path.exists('net.pth'):
        net.load_state_dict(torch.load('net.pth'))
        net.eval()
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            # batch_sampler=BatchSampler(
            #     WeightedRandomSampler(weights=train_weights,
            #                           num_samples=len(dataset)),
            batch_sampler=BatchSampler(
                RandomSampler(data_source=dataset),
                batch_size=batch_size,
                drop_last=True),
            # num_workers=config['workers']
        )
        # dataloader = torch.utils.data.DataLoader(
        #     dataset,
        #     # batch_sampler=BatchSampler(
        #     #     WeightedRandomSampler(weights=train_weights,
        #     #                           num_samples=len(dataset)),
        #     # batch_sampler=BatchSampler(
        #     #     RandomSampler(data_source=dataset),
        #     batch_size=batch_size,
        #         # drop_last=True),
        #     # num_workers=config['workers']
        # )
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for i in range(n_epochs):
            for j, sample in enumerate(dataloader):
                optimizer.zero_grad()
                x = sample['points'].float()
                x = x.view(batch_size, 3000)
                t = x
                p = net(x)
                loss = criterion(p, t)
                loss.backward()
                optimizer.step()
                print(loss)
    # params = list(net.parameters())
    torch.save(net.state_dict(), 'net.pth')
    # x2
    # with torch.no_grad():
    # x2 = net.fc3(torch.randn(3, 256))
    x2 = net.fc3(torch.from_numpy(np.array([1. if x == 0 else 0. for x in range(256)])).float())
    x2 = net.fc4(x2)
    print(x2)


if __name__ == '__main__':
    composed = torchvision.transforms.Compose([ToSpherical(), Normalize()])
    ds = Dataset2('/home/romanzes/Programs/pointnet-pytorch/s3d_rooms/test/rooms/office_25.txt',
                  transform=composed)
    train(ds)
    # plot_2('/home/romanzes/Programs/pointnet-pytorch/s3d_rooms/test/rooms/office_25.txt')