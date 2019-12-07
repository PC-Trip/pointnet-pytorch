# -*- coding: utf-8 -*-

import argparse
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import lera

from datasets import RoomsDataset_mk2
from pointnet import PointNetSeg
from utils import get_s3d_num2cat

def get_path_of_last_model(config):
    models_path = os.path.join(config['root'], config['modelsFolder'])
    files = list(filter(lambda f: os.path.isfile(os.path.join(models_path, f)) and f.endswith('.pth'), os.listdir(models_path) ))
    if len(files) == 0:
        return None, 0
    files.sort(key=lambda f: int(f.split('.')[0].split('_')[-1] ))
    return os.path.join(models_path, files[-1]), int(files[-1].split('.')[0].split('_')[-1])

def train(config):
    print('Random seed: %d' % int(config['seed']))
    torch.manual_seed(config['seed'])
    print("Training {} epochs".format(config['nepochs']))
    
    torch.backends.cudnn.benchmark = True
    dataset = RoomsDataset_mk2(path=os.path.join(config['root'], 'train', 'rooms'), size=10000, npoints=config['npoints'])
    test_dataset = RoomsDataset_mk2(path=os.path.join(config['root'], 'test', 'rooms'), size=1000, npoints=config['npoints'])

    print('training s3dis')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batchsize'], shuffle=True, 
                num_workers=config['workers'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batchsize'], shuffle=True, 
        num_workers=config['workers'])

    num2cat = get_s3d_num2cat()
    num_classes = len(list(num2cat))
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
                points, labels = data
                points = points.transpose(2, 1)
                points, labels = points.to(device), labels.to(device)
                optimizer.zero_grad()
                classifier = classifier.train()
                pred, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                labels = labels.view(-1, 1)[:, 0]
                loss = F.nll_loss(pred, labels)
                loss.backward()
                optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(labels.data).cpu().sum()
                train_acc = correct.item() / float(config['batchsize']*config['npoints'])
                train_iou = correct.item() / float(2*config['batchsize']*config['npoints']-correct.item())
                print('epoch %d: %d/%d | train loss: %f | train acc: %f | train iou: %f' % (epoch+1, i+1, num_batch+1, loss.item(), train_acc, train_iou))
                train_acc_epoch.append(train_acc)
                train_iou_epoch.append(train_iou)
                # lera.log({
                #     'train loss': loss.item(), 
                #     'train acc': train_acc, 
                #     'train IoU': train_iou}
                #     )

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
                    print(blue('epoch %d: %d/%d | test loss: %f | test acc: %f | test iou: %f') % (epoch+1, i+1, num_batch+1, loss.item(), test_acc, test_iou))
                    test_acc_epoch.append(test_acc)
                    test_iou_epoch.append(test_iou)
                    # lera.log({
                    #     'test loss': loss.item(), 
                    #     'test acc': test_acc, 
                    #     'test IoU': test_iou})
            print(yellow('epoch %d | mean train acc: %f | mean train IoU: %f') % (epoch+1, np.mean(train_acc_epoch), np.mean(train_iou_epoch)))
            print(red('epoch %d | mean test acc: %f | mean test IoU: %f') % (epoch+1, np.mean(test_acc_epoch), np.mean(test_iou_epoch)))
            # lera.log({
            #     'mean train acc': np.mean(train_acc_epoch), 
            #     'mean train iou': np.mean(train_iou_epoch), 
            #     'mean test acc': np.mean(test_acc_epoch), 
            #     'mean test iou': np.mean(test_iou_epoch)})
        except KeyboardInterrupt:
            print('User interruption')
            break
        
        finally:
            torch.save(classifier.state_dict(), os.path.join(config['modelsFolder'], '{}_model_{}.pth'.format(config['dataset'], model_epoch_cumulatiove_base+epoch)))


if __name__ == '__main__':
    config = {
        'root': '.',
        'npoints': 4096,
        'dataset': 's3dis',
        'seed': 42,
        'batchsize': 30,
        'workers': 0,
        'modelsFolder': 'modelsFolder',
        'lr': 0.01,
        'momentum': 0.9,
        'classname': '',
        'nepochs': 1,
        'model': None,
        'continue': True,
    }

    train(config)


