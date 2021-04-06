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

import matplotlib.pyplot as plt
import torchgan
import torchgan.layers

from classes import MMParser, Generator_V,Generator, Discriminator_S, Discriminator



ngpu=2
image_size=128
nz=100
epochs=875

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
noise = torch.randn(image_size, nz, 1, 1, device=device)
netG=Generator(ngpu).to(device)
for i in range(epochs):
	netG.load_state_dict(torch.load("wgan_FL128_epochs50.pth"))
	fake = netG(noise)
	torch.save(fake,"/scratch/trahman2/molecules_data/data_128/wgan_epoch50_"+str(i))




