import open3d as o3d
import numpy as np
from matplotlib import cm
import torch
from pointnet import PointNetSeg
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from deco import pc_normalize, ps_to_spherical


@pc_normalize(norm_type='spherical', center=False, verbose=False)
@ps_to_spherical(verbose=False)
def transform(points, labels):
    return points, labels


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False,
             hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (
            columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (
                len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


class Callback:
    def __init__(self, point_to_class_map,
                 pred_point_to_class_map, classes_names):
        self.points = set()
        self.ptc_map = point_to_class_map
        self.pred_ptc_map = pred_point_to_class_map
        self.cs_ns = classes_names

    def __call__(self, vis):
        idxs = vis.get_picked_points()
        for i in idxs:
            if i not in self.points:
                self.points.add(i)
                pc = self.ptc_map[i]
                pn = self.cs_ns[pc]
                pred_pc = self.pred_ptc_map[i]
                pred_pn = self.cs_ns[pred_pc]
                print('{} point class: {} - {}, predicted: {} - {}'.format(
                    i, pc, pn, pred_pc, pred_pn))
        return False


if __name__ == '__main__':
    file_name = 's3d2/test/rooms/WC_1.txt'
    print(file_name)
    data = np.loadtxt(file_name)
    labels_names = [x.split()[0] for x in open('s3d_cat2num.txt')]
    print(labels_names)
    points, colors, labels = data[:, :3].astype(
                np.float32), data[:, 3:6] / 255, data[:, 6].astype(np.long)
    print(points)
    print(colors)
    print(labels)
    n_points = points.shape[0]
    print('n_points: {}'.format(n_points))
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

    config = {
        'root': 'Stanford3dDataset_v1.2',
        'npoints': 4096,
        'dataset': 's3disneisphe',
        'seed': 43,
        'batchsize': 25,
        'workers': 2,
        'outf': 'outFolderS',
        'lr': 0.01,
        'momentum': 0.9,
        'classname': '',
        'nepochs': 10,
        'model': 's3disneisphe_model_0.pth',
        'continue': True,
        'verbose': 1,
    }

    # print('Random seed: %d' % int(config['seed']))
    # torch.manual_seed(config['seed'])
    # print("Training {} epochs".format(config['nepochs']))
    #
    # torch.backends.cudnn.benchmark = True
    # if config['dataset'] == 's3dis':
    #     dataset = S3dDataset(root=config['root'], npoints=config['npoints'], train=True)
    #     test_dataset = S3dDataset(root=config['root'], npoints=config['npoints'], train=False)
    # else:
    #     dataset = S3dDatasetNeiSphe(root=config['root'], npoints=config['npoints'], train=True)
    #     test_dataset = S3dDatasetNeiSphe(root=config['root'], npoints=config['npoints'], train=False)
    # print('training s3dis')
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=config['batchsize'],
    #                                          shuffle=True,
    #                                          num_workers=config['workers'],
    #                                          drop_last=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batchsize'], shuffle=True,
    #     num_workers=config['workers'], drop_last=True)

    # num_classes = dataset.num_classes
    # print('number of classes: %d' % num_classes)
    # print('train set size: %d | test set size: %d' % (len(dataset), len(test_dataset)))
    # try:
    #     os.makedirs(config['outf'])
    # except:
    #     pass

    # blue = lambda x: '\033[94m' + x + '\033[0m'
    # yellow = lambda x: '\033[93m' + x + '\033[0m'
    # red = lambda x: '\033[91m' + x + '\033[0m'
    num_classes = 14
    classifier = PointNetSeg(k=num_classes)

    model_epoch_cumulatiove_base = 0
    # if config.get('model'):
    print('Loading model from: {}'.format(config.get('model')))
    classifier.load_state_dict(torch.load(config['model']))
    # elif config.get('continue'):
    # model_path, model_epoch_cumulatiove_base = get_path_of_last_model(config)
    # if model_path:
    #     print('Loading model from: {}'.format(model_path))
    #     classifier.load_state_dict(torch.load(model_path))
    # optimizer = optim.SGD(classifier.parameters(), lr=config['lr'], momentum=config['momentum'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    # if config.get('mgpu'):
    #     classifier = torch.nn.DataParallel(classifier, device_ids=config['gpuids'])

    # num_batch = len(dataset) / config['batchsize']

    cnt = 0
    n_batches = int(n_points / 4096)
    batches_per_step = 20
    print(n_batches)
    n_steps = int(n_batches / batches_per_step)
    print(n_steps)
    predicted = []
    for i in range(n_steps):
        idx0 = i * 4096 * batches_per_step
        idx1 = (i + 1) * 4096 * batches_per_step
        print('step {}/{}, points from {} to {}'.format(i + 1, n_steps, idx0, idx1))
        batch_points = torch.from_numpy(points[idx0:idx1].reshape(-1, 4096, 3))
        for j, bp in enumerate(batch_points):
            ps, _ = transform(bp, 'a')
            batch_points[j] = ps
        batch_points = batch_points.transpose(2, 1)
        batch_points = batch_points.to(device)
        classifier = classifier.eval()
        with torch.no_grad():
            pred, _ = classifier(batch_points)
        pred = pred.view(-1, num_classes)
        pred_choice = pred.data.max(1)[1]
        predicted.append(pred_choice.numpy())
    pred_labels = np.array(predicted).flatten()
    true_labels = labels[:len(pred_labels)]
    # Report + Confusion Matrix
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)
    all_labels = np.unique(
        np.concatenate((unique_true_labels, unique_pred_labels)))
    step_names = [labels_names[x] for x in all_labels]
    print(classification_report(true_labels, pred_labels,
                                target_names=step_names))
    m = confusion_matrix(true_labels, pred_labels)
    print_cm(m, step_names)

    # cmap = cm.get_cmap('tab10')
    # n_labels = len(unique_pred_labels)
    # if n_labels > 1:
    #     labels_colors = np.apply_along_axis(
    #         lambda x: cmap(x / max_label), 0, pred_labels)[:, :3]
    # else:
    #     labels_colors = np.full_like(pred_labels, cmap(0)[:-1])
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:len(pred_labels)])
    # pcd.colors = o3d.utility.Vector3dVector(labels_colors)
    # vis = o3d.visualization.VisualizerWithEditing()
    # c = Callback(true_labels, pred_labels, labels_names)
    # vis.register_animation_callback(c)
    # vis.create_window(width=500, height=500,
    #                   left=50, top=50)
    # vis.add_geometry(pcd)
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([0, 0, 0])
    # vis.run()
    # vis.destroy_window()

    # cmap = cm.get_cmap('tab10')
    # if n_labels > 1:
    #     labels_colors = np.apply_along_axis(
    #         lambda x: cmap(x / max_label), 0, labels)[:, :3]
    # else:
    #     labels_colors = np.full_like(labels, cmap(0)[:-1])
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(labels_colors)
    # vis = o3d.visualization.VisualizerWithEditing()
    # c = Callback(labels, labels_names)
    # vis.register_animation_callback(c)
    # vis.create_window(width=500, height=500,
    #                   left=50, top=50)
    # vis.add_geometry(pcd)
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([0, 0, 0])
    # vis.run()
    # vis.destroy_window()
