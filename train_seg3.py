# -*- coding: utf-8 -*-

import argparse
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from datasets import RoomsDataset_mk2
from pointnet import PointNetSeg
from utils import get_s3d_num2cat
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_conf_matrix(conf_mtrx, labels, file_path='temp_conf_mtrx.png'):
    df_cm = pd.DataFrame(np.array(conf_mtrx), index=labels, columns=labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(file_path)
    plt.close()


def get_path_of_last_model(config):
    models_path = os.path.join(config['root'], config['modelsFolder'])
    files = list(filter(lambda f: os.path.isfile(os.path.join(models_path, f)) and f.endswith('.pth'), os.listdir(models_path) ))
    if len(files) == 0:
        return None, 0
    files.sort(key=lambda f: int(f.split('.')[0].split('_')[-1] ))
    return os.path.join(models_path, files[-1]), int(files[-1].split('.')[0].split('_')[-1])



def train(config):
    mlflow.log_params(config)

    print('Random seed: %d' % int(config['seed']))
    torch.manual_seed(config['seed'])
    print("Training {} epochs".format(config['nepochs']))
    
    torch.backends.cudnn.benchmark = True

    dataset = RoomsDataset_mk2(path=os.path.join(config['root'], 'train', 'conferences'), size=10000, npoints=config['npoints'])
    test_dataset = RoomsDataset_mk2(path=os.path.join(config['root'], 'test', 'conferences'), size=100, npoints=None)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batchsize'], shuffle=True, 
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
        classifier.load_state_dict(torch.load())
        mlflow.log_param('start', config['model'])
    elif config.get('continue'):
        model_path, model_epoch_cumulatiove_base = get_path_of_last_model(config)
        if model_path:
            print('Loading model from: {}'.format(model_path))
            mlflow.log_param('start', model_path)
            classifier.load_state_dict(torch.load(model_path))


    optimizer = optim.Adam(classifier.parameters(), lr=config['lr'])
    # optimizer = optim.SGD(classifier.parameters(), lr=config['lr'], momentum=config['momentum'])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.97)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    if config.get('mgpu'):
        classifier = torch.nn.DataParallel(classifier, device_ids=config['gpuids'])

    num_batch = len(dataset) / config['batchsize']

    for epoch in range(config['nepochs']):
        train_acc_epoch, train_iou_epoch = [], []
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
                cnf_mtrx = confusion_matrix(labels.cpu().tolist(), pred_choice.cpu().tolist(), labels=sorted(list(num2cat)))
                conf_mtrx_file_path = os.path.join("temp", f"cnf_mtrx_{epoch}_{i}.png")
                plot_conf_matrix(cnf_mtrx, [num2cat[num] for num in sorted(list(num2cat))], conf_mtrx_file_path)

                encountered_class_nums = np.unique( np.array(labels.cpu().tolist()) )
                prec, recall, f1, _ = precision_recall_fscore_support(labels.cpu().tolist(), pred_choice.cpu().tolist(), labels=sorted(encountered_class_nums))
                for i, num in enumerate(encountered_class_nums):
                    name = num2cat[num]
                    mlflow.log_metrics({
                        f"{name}_acc": prec[i],
                        f"{name}_recall": recall[i],
                        f"{name}_f1": f1[i],
                    })

                mlflow.log_artifact(conf_mtrx_file_path)
                mlflow.log_metric('train_acc', train_acc, step=1)
                mlflow.log_metric('train_iou', train_iou, step=1)
                mlflow.log_metric('train_iou', train_iou, step=1)
                mlflow.log_metric('lr', get_lr(optimizer), step=1)

                train_acc_epoch.append(train_acc)
                train_iou_epoch.append(train_iou)
                scheduler.step()

                if (i+1) % 20 == 0:
                    test_acc_epoch, test_iou_epoch = [], []
                    test_dataset.reset()
                    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batchsize'], shuffle=True, num_workers=config['workers'])
                    
                    labels_for_stat = []
                    preds_for_stat = []
                    for j in range(test_dataset.size):
                        points, labels = test_dataset[j]
                        points, labels = torch.unsqueeze(torch.from_numpy(points), 0), torch.unsqueeze(torch.from_numpy(labels), 0)
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
                        print(blue('epoch %d: %d/%d | test loss: %f | test acc: %f | test iou: %f') % (epoch+1, j+1, test_dataset.size+1, loss.item(), test_acc, test_iou))
                        test_acc_epoch.append(test_acc)
                        test_iou_epoch.append(test_iou)
                        labels_for_stat.append(np.array(labels.cpu().tolist()))
                        preds_for_stat.append(np.array(pred_choice.cpu().tolist()))

                    all_labels = np.concatenate(labels_for_stat)
                    all_preds = np.concatenate(preds_for_stat)
                    encountered_class_nums = np.unique( all_labels )
                    prec, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=sorted(encountered_class_nums))
                    for k, num in enumerate(encountered_class_nums):
                        name = num2cat[num]
                        mlflow.log_metrics({
                            f"{name}_acc": prec[k],
                            f"{name}_recall": recall[k],
                            f"{name}_f1": f1[k],
                        })

                    mlflow.log_metric('test_acc', np.mean(test_acc_epoch), step=1)
                    mlflow.log_metric('test_iou', np.mean(test_iou), step=1)
                    print(red('epoch %d | mean test acc: %f | mean test IoU: %f') % (epoch+1, np.mean(test_acc_epoch), np.mean(test_iou_epoch)))
                
            print(yellow('epoch %d | mean train acc: %f | mean train IoU: %f') % (epoch+1, np.mean(train_acc_epoch), np.mean(train_iou_epoch)))

        except KeyboardInterrupt:
            print('User interruption')
            break
        
        finally:
            model_path = os.path.join(config['modelsFolder'], '{}_model_{}.pth'.format(config['dataset'], model_epoch_cumulatiove_base+epoch))
            print("saving to {}".format(model_path))
            torch.save(classifier.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            mlflow.log_param('ended', model_path)

if __name__ == '__main__':
    config = {
        'root': '.',
        'npoints': 4096,
        'dataset': 's3dis',
        'seed': 42,
        'batchsize': 10,
        'workers': 0,
        'modelsFolder': 'modelsFolder',
        'lr': 0.001,
        'momentum': 0.1,
        'classname': '',
        'nepochs': 30,
        'model': None,
        'continue': True,
    }

    train(config)


