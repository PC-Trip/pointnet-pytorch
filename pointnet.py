# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

input_dim = 6 # 3

class STN3D(nn.Module):
	def __init__(self):
		super(STN3D, self).__init__()
		self.conv1 = nn.Conv1d(input_dim, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 1024, 1)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, input_dim*input_dim)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)
		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)


	def forward(self, x):
		batchsize = x.size()[0]
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)

		x = F.relu(self.bn4(self.fc1(x)))
		x = F.relu(self.bn5(self.fc2(x)))
		x = self.fc3(x)

		# iden = autograd.Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize, 1)
		iden = autograd.Variable(torch.from_numpy(np.eye(input_dim).astype(np.float32).reshape(1, -1))).view(1,input_dim**2).repeat(batchsize, 1)
		if x.is_cuda:
			iden = iden.cuda()
		x += iden
		x = x.view(-1, input_dim, input_dim)
		return x


class PointNetfeat(nn.Module):
	def __init__(self, global_feat=True):
		super(PointNetfeat, self).__init__()
		self.stn = STN3D()
		self.conv1 = nn.Conv1d(input_dim, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 1024, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)
		self.global_feat = global_feat


	def forward(self, x):
		batchsize = x.size()[0]
		n_pts = x.size()[2]
		trans = self.stn(x)
		x = x.transpose(2, 1)
		x = torch.bmm(x, trans)
		x = x.transpose(2, 1)
		x = F.relu(self.bn1(self.conv1(x)))
		pointfeat = x
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)
		if self.global_feat:
			return x, trans
		else:
			x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
			return torch.cat([x, pointfeat], 1), trans


class PointNetCls(nn.Module):
	def __init__(self, k=2):
		super(PointNetCls, self).__init__()
		self.k = k
		self.feat = PointNetfeat(global_feat=True)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, k)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)


	def forward(self, x):
		x, trans = self.feat(x)
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1), trans


class PointNetSeg(nn.Module):
	def __init__(self, k=2):
		super(PointNetSeg, self).__init__()
		self.k = k
		self.feat = PointNetfeat(global_feat=False)
		self.conv1 = nn.Conv1d(1088, 512, 1)
		self.conv2 = nn.Conv1d(512, 256, 1)
		self.conv3 = nn.Conv1d(256, 128, 1)
		self.conv4 = nn.Conv1d(128, self.k, 1)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)
		self.bn3 = nn.BatchNorm1d(128)

	
	def forward(self, x):
		batchsize = x.size()[0]
		n_pts = x.size()[2]
		x, trans = self.feat(x)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = self.conv4(x)
		x = x.transpose(2, 1).contiguous()
		x = F.log_softmax(x.view(-1, self.k), dim=-1)
		x = x.view(batchsize, n_pts, self.k)
		return x, trans


if __name__ == '__main__':

	sim_data = autograd.Variable(torch.randn(32, 3, 2048))
	trans = STN3D()
	out = trans(sim_data)
	print('stn', out.size())

	pointfeat = PointNetfeat(global_feat=True)
	out, _ = pointfeat(sim_data)
	print('global feat', out.size())

	pointfeat = PointNetfeat(global_feat=False)
	out, _ = pointfeat(sim_data)
	print('point feat', out.size())

	cls = PointNetCls(k=4)
	out, _ = cls(sim_data)
	print('class', out.size())

	seg = PointNetSeg(k=4)
	out, _ = seg(sim_data)
	print('seg', out.size())

























































