# -*- coding: utf-8 -*-

import argparse
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler

# import lera
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from pprint import pprint

from datasets2 import S3dDataset
from pointnet import PointNetSeg


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


def get_path_of_last_model(config):
    models_path = os.path.join(config['root'], config['outf'])
    files = list(filter(lambda f: os.path.isfile(os.path.join(models_path, f)) and f.endswith('.pth'), os.listdir(models_path) ))
    if len(files) == 0:
        return None, 0
    files.sort(key=lambda f: int(f.split('.')[0].split('_')[-1] ))
    return os.path.join(models_path, files[-1]), int(files[-1].split('.')[0].split('_')[-1])


def log_metric_to_mlflow(true, pred, ds_type, target_names=None,
                         is_cm=True, cm_plot=True, cm_norm=None,
                         verbose=1, root='.', ds_name='dataset'):
    true_labels = np.unique(true)
    pred_labels = np.unique(pred)
    step_labels = np.unique(np.concatenate((true_labels, pred_labels)))
    if target_names is None:
        target_names = map(str, step_labels)
    step_names = [target_names[x] for x in step_labels]
    cr = classification_report(true, pred, target_names=step_names,
                               output_dict=True)
    for label, metrics in cr.items():
        if isinstance(metrics, dict):
            for m, v in metrics.items():
                mlflow.log_metric('{}_cr_{}_{}'.format(ds_type, label, m), v)
        else:
            mlflow.log_metric('{}_cr_{}'.format(ds_type, label), metrics)
    if verbose > 0:
        pprint(cr)
    if is_cm:
        if cm_norm is None:
            norms = ['true', 'pred', 'all']
        else:
            norms = [cm_norm]
        for norm in norms:
            cmat = confusion_matrix(true, pred, normalize=norm)
            for i, li in enumerate(step_names):
                for j, lj in enumerate(step_names):
                    mlflow.log_metric('{}_cm_{}_{}_{}'.format(
                        ds_type, li, lj, norm), cmat[i, j])
            if cm_plot:
                # Save cm image
                dir_name = 'plots'
                dir_path = os.path.join(root, dir_name)
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                cm_file_name = os.path.join(
                    dir_path, '{}_{}_cm_{}_{}.png'.format(
                        ds_name, ds_type, norm, time.strftime("%Y%m%d-%H%M%S")))
                fig, ax = plt.subplots()
                im = ax.imshow(cmat, cmap=cm.get_cmap('Blues'))
                # We want to show all ticks...
                ax.set_xticks(np.arange(len(step_names)))
                ax.set_yticks(np.arange(len(step_names)))
                # ... and label them with the respective list entries
                ax.set_xticklabels(step_names)
                ax.set_yticklabels(step_names)
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
                # Loop over data dimensions and create text annotations.
                for i in range(len(step_names)):
                    for j in range(len(step_names)):
                        v = int(round(cmat[i, j]*100))
                        text = ax.text(j, i, '{}'.format(v), ha="center",
                                       va="center",
                                       color="w" if v > 20 else 'k')
                ax.set_title("Confusion matrix")
                fig.tight_layout()
                plt.savefig(cm_file_name, dpi=300)
                mlflow.log_artifact(cm_file_name)
        if verbose > 0:
            print_cm(cmat, step_names)


def get_weights(dataset, ds_type, n_classes, root='.', exist=True):
    weights_fn = os.path.join(root, '{}_ds_weights.pt'.format(ds_type))
    if not os.path.exists(weights_fn):
        print('Evaluating {} weights'.format(ds_type))
        n_samples = len(dataset)
        counts = torch.zeros(n_samples, n_classes).long()
        for i, (k, v) in enumerate(dataset.labels.items()):
            n_points = len(dataset.points[k])
            print('{}/{} {} {}'.format(i + 1, n_samples, n_points, k))
            if isinstance(v, int):
                u, c = torch.tensor([v]), torch.tensor([n_points])
            else:
                u, c = v.unique(return_counts=True)
            # print(u, c)
            counts[i].scatter_(0, u, c)
        # print(counts)
        if exist:
            cols_counts_sum = torch.from_numpy(
                np.count_nonzero(counts.numpy(), axis=0))
        else:
            cols_counts_sum = counts.sum(0)
        print(cols_counts_sum)
        zeros = (cols_counts_sum == 0).nonzero()  # WBD
        if zeros.size()[0] > 0:
            if ds_type == 'train':
                raise ValueError('Labels {} are not presented '
                                 'in datatset!'.format(zeros))
            elif ds_type == 'test':
                print('Warning! Labels {} are not presented '
                      'in datatset!'.format(zeros))
        cols_weights = cols_counts_sum.sum().float() / cols_counts_sum
        cols_weights[cols_weights == float("Inf")] = 0
        # print(cols_weights)
        cols_weights = cols_weights / cols_weights.sum()
        # print(cols_weights)
        # print(cols_weights.sum())
        rows_counts_sum = counts.sum(1)
        # print(rows_counts_sum)
        rows_weights = counts / rows_counts_sum.view(-1, 1).float()
        # print(rows_weights)
        weights = rows_weights * cols_weights
        # print(weights)
        weights = weights.sum(1)
        # print(weights)
        weights = weights / weights.sum()
        # print(weights)
        print('Saving {}'.format(weights_fn))
        torch.save(weights, weights_fn)
    else:
        print('Loading {}'.format(weights_fn))
        weights = torch.load(weights_fn)
    print(weights)
    print(weights.sum())
    return weights


def train(config):
    mlflow.set_experiment(config['dataset'])
    mlflow.log_params(config)
    print('Random seed: %d' % int(config['seed']))
    torch.manual_seed(config['seed'])
    print("Training {} epochs".format(config['nepochs']))
    torch.backends.cudnn.benchmark = True
    dataset = S3dDataset(root=config['root'], npoints=config['npoints'], train=True, load=True)
    test_dataset = S3dDataset(root=config['root'], npoints=config['npoints'], train=False, load=True)
    num_classes = dataset.num_classes
    if config['balance']:
        train_weights = get_weights(dataset, 'train', root=config['root'],
                                    n_classes=num_classes)
        test_weights = get_weights(test_dataset, 'test', root=config['root'],
                                   n_classes=num_classes)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=BatchSampler(
                WeightedRandomSampler(weights=train_weights,
                                      num_samples=len(dataset)),
                batch_size=config['batchsize'],
                drop_last=True),
            num_workers=config['workers'])
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_sampler=BatchSampler(
                WeightedRandomSampler(weights=test_weights,
                                      num_samples=len(test_dataset)),
                batch_size=config['batchsize'],
                drop_last=True),
            num_workers=config['workers'])
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batchsize'], shuffle=True,
                    num_workers=config['workers'], drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batchsize'], shuffle=True,
            num_workers=config['workers'], drop_last=True)
    print('number of classes: %d' % num_classes)
    print('train set size: %d | test set size: %d' % (len(dataset), len(test_dataset)))
    try:
        os.makedirs(config['outf'])
    except:
        pass
    blue = lambda x: '\033[94m' + x + '\033[0m'
    yellow = lambda x: '\033[93m' + x + '\033[0m'
    red = lambda x: '\033[91m' + x + '\033[0m'
    classifier = PointNetSeg(k=num_classes)
    model_epoch_cumulatiove_base = 0
    if config.get('model'):
        print('Loading model from: {}'.format(config.get('model')))
        classifier.load_state_dict(torch.load(config['model']))
    elif config.get('continue'):
        model_path, model_epoch_cumulatiove_base = get_path_of_last_model(config)
        if model_path:
            print('Loading model from: {}'.format(model_path))
            classifier.load_state_dict(torch.load(model_path))
        # model_path_dir = ...
        # run_id = "96771d893a5e46159d9f3b49bf9013e2"
        # pytorch_model = mlflow.pytorch.load_model(
        #     "runs:/" + run_id + "/" + model_path_dir)
    optimizer = optim.SGD(classifier.parameters(), lr=config['lr'], momentum=config['momentum'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    if config.get('mgpu'):
        classifier = torch.nn.DataParallel(classifier, device_ids=config['gpuids'])
    num_batch = len(dataset) / config['batchsize']
    for epoch in range(config['nepochs']):
        train_acc_epoch, train_iou_epoch, test_acc_epoch, test_iou_epoch = [], [], [], []
        try:
            for i, data in enumerate(dataloader):
                t0 = time.time()
                points, labels = data
                points = points.transpose(2, 1)
                points, labels = points.to(device), labels.to(device)
                optimizer.zero_grad()
                classifier = classifier.train()
                pred, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                labels = labels.view(-1, 1)[:, 0]
                if config['weight']:
                    # unique labels, counts
                    u, c = torch.unique(labels, return_counts=True)
                    print(u, c)
                    w = c.sum().float() / c  # weights
                    w = w / w.sum()  # normalized weights
                    # filling missing labels weights with zeros
                    w = torch.zeros(num_classes).scatter_(0, u, w)
                    print(w)
                    loss = F.nll_loss(pred, labels, weight=w)
                else:
                    loss = F.nll_loss(pred, labels)
                loss.backward()
                optimizer.step()
                time_per_step = time.time() - t0
                t0 = time.time()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(labels.data).cpu().sum()
                train_acc = correct.item() / float(config['batchsize']*config['npoints'])
                train_iou = correct.item() / float(2*config['batchsize']*config['npoints']-correct.item())
                train_acc_epoch.append(train_acc)
                train_iou_epoch.append(train_iou)
                mlflow.log_metric('train_acc', train_acc)
                mlflow.log_metric('train_iou', train_iou)
                mlflow.log_metric('train_loss',  loss.item())
                log_metric_to_mlflow(
                    labels.data.numpy(), pred_choice.numpy(), 'train',
                    target_names=['board', 'floor', 'door', 'bookcase',
                                  'column', 'ceiling', 'wall', 'stairs',
                                  'beam', 'chair', 'clutter', 'table',
                                  'window', 'sofa'],
                    cm_plot=False, cm_norm='true',
                    root=config['root'], ds_name=config['dataset'], verbose=0)
                time_per_log = time.time() - t0
                mlflow.log_metric('time_per_step',  time_per_step)
                mlflow.log_metric('time_per_log',  time_per_log)
                print('epoch %d: %d/%d | train loss: %f | train acc: %f | train iou: %f' % (epoch+1, i+1, num_batch+1, loss.item(), train_acc, train_iou))
                if (i+1) % 10 == 0:
                    j, data = next(enumerate(test_dataloader, 0))
                    points, labels = data
                    points = points.transpose(2, 1)
                    points, labels = points.to(device), labels.to(device)
                    classifier = classifier.eval()
                    with torch.no_grad():
                        pred, _ = classifier(points)
                    pred = pred.view(-1, num_classes)
                    labels = labels.view(-1, 1)[:, 0]
                    loss = F.nll_loss(pred, labels)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(labels.data).cpu().sum()
                    test_acc = correct.item() / float(config['batchsize']*config['npoints'])
                    test_iou = correct.item() / float(2*config['batchsize']*config['npoints']-correct.item())
                    test_acc_epoch.append(test_acc)
                    test_iou_epoch.append(test_iou)
                    mlflow.log_metric('test_acc', test_acc)
                    mlflow.log_metric('test_iou', test_iou)
                    mlflow.log_metric('test_loss', loss.item())
                    mlflow.log_metric('train_acc', train_acc)
                    mlflow.log_metric('train_iou', train_iou)
                    mlflow.log_metric('train_loss', loss.item())
                    log_metric_to_mlflow(
                        labels.data.numpy(), pred_choice.numpy(), 'test',
                        target_names=['board', 'floor', 'door', 'bookcase',
                                      'column', 'ceiling', 'wall', 'stairs',
                                      'beam', 'chair', 'clutter', 'table',
                                      'window', 'sofa'], cm_plot=True,
                        root=config['root'], ds_name=config['dataset'],
                        verbose=0)
                    # mlflow.pytorch.log_model(classifier, 'model')  # mlfow has no attribute 'pytorch'
                    # mlflow.pytorch.save_model(classifier, os.path.join(
                    #     config['outf'], '{}_model_{}.pth'.format(
                    #         config['dataset'],model_epoch_cumulatiove_base + epoch))
                    torch.save(classifier.state_dict(),
                               os.path.join(config['outf'],
                                            '{}_model_{}.pth'.format(
                                                config['dataset'],
                                                model_epoch_cumulatiove_base + epoch)))
                    print(blue('epoch %d: %d/%d | test loss: %f | test acc: %f | test iou: %f') % (epoch+1, i+1, num_batch+1, loss.item(), test_acc, test_iou))
            print(yellow('epoch %d | mean train acc: %f | mean train IoU: %f') % (epoch+1, np.mean(train_acc_epoch), np.mean(train_iou_epoch)))
            print(red('epoch %d | mean test acc: %f | mean test IoU: %f') % (epoch+1, np.mean(test_acc_epoch), np.mean(test_iou_epoch)))
        except KeyboardInterrupt:
            print('User interruption')
            break
        finally:
            torch.save(classifier.state_dict(), os.path.join(config['outf'], '{}_model_{}.pth'.format(config['dataset'], model_epoch_cumulatiove_base+epoch)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--seed', type=int, help='random seed')
    # parser.add_argument('-dset', '--dataset', type=str, required=True, help='dataset to train on, one of modelnet, shapenet16 and s3dis')
    # parser.add_argument('-c', '--classname', type=str, default='Chair', help='one of 16 categories on shapenet16')
    # parser.add_argument('-r', '--root', type=str, required=True, help='path to dataset')
    # parser.add_argument('-np', '--npoints', type=int, help='number of points to sample')
    # parser.add_argument('-bs', '--batchsize', type=int, default=32, help='batch size')
    # parser.add_argument('-ws', '--workers', type=int, default=4, help='number of workers')
    # parser.add_argument('-out', '--outf', type=str, default='./checkpoints', help='path to save model checkpoints')
    # parser.add_argument('--model', type=str, default='', help='checkpoint dir')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum in SGD')
    parser.add_argument('--mgpu', type=bool, default=False, help='whether to utilize multiple gpus')
    parser.add_argument('--gpuids', nargs='+', type=int, help='which gpus to use')
    # parser.add_argument('--nepochs', type=int, default=100, help='epochs to train')
    # config = parser.parse_args()

    config = {
        'root': 'Stanford3dDataset_v1.2',
        'npoints': 4096,
        'dataset': 's3disw',
        'seed': 42,
        'batchsize': 300,
        'workers': 0,  # FIXME Memory Leak when workers > 0! https://github.com/pytorch/pytorch/issues/13246
        'outf': 'outFolder',
        'lr': 0.01,
        'momentum': 0.9,
        'classname': '',
        'nepochs': 10,
        'model': None,
        'continue': True,
        'verbose': 1,
        'weight': True,
        'balance': True
    }

    train(config)


