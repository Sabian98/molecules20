
import numpy as np
import random
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import spectral_norm
import time
import matplotlib.pyplot as plt
import torchgan
import torchgan.layers

from torchgan.layers import VirtualBatchNorm

class MMParser(Dataset):
	def __init__(self,arr):
		self.arr=arr
	def __len__(self):
		return len(self.arr)
	def __getitem__(self, idx):
		return self.arr[idx]

#generator class
class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			nn.ConvTranspose2d(100, 512, kernel_size=4, stride=4, padding=0, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, padding=0, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
		)

	def forward(self, input):
		return self.main(input)


class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			nn.Conv2d(128, 256, kernel_size=4, stride=4, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
			nn.Sigmoid(),
		)

	def forward(self, input):
		return self.main(input)



class Generator_V(nn.Module):
	def __init__(self, ngpu):
		super(Generator_V, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			nn.ConvTranspose2d(100, 512, kernel_size=4, stride=4, padding=0, bias=False),
			VirtualBatchNorm(512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
			VirtualBatchNorm(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, padding=0, bias=False),
			VirtualBatchNorm(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
			VirtualBatchNorm(64),
			nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
		)

	def forward(self, input):
		return self.main(input)


class Discriminator_S(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator_S, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=4, padding=1, bias=False)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)),
			nn.Sigmoid(),
		)

	def forward(self, input):
		return self.main(input)
