
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

from classes import MMParser, Generator_V,Generator, Discriminator_S, Discriminator
#

scaled_factor=10

def weights_init(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1 and classname.find('VirtualBatchNorm') == -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



f=open("/scratch/trahman2/train_idx.txt")
fr=f.readlines()
f.close()
id_arr=[]
for lines in fr:
	cont=lines.split(" ")
	if(int(cont[1])==128 ):
		id_arr.append(cont[0])
maps_arr=[]
scaled_map=[]

for id in id_arr:

	f_cont=np.loadtxt("/scratch/trahman2/map_128_1/"+id+".txt")
	maps_arr.append(f_cont)


for map in maps_arr:
	# map=map*(1/scaled_factor)
	scaled_map.append(map)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
		ngpu = 2
	
# Set random seed for reproducibility
manualSeed = 666
# manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128
image_size=128
num_epochs=110

lr=10e-4
lr2=10e-3

beta1=0.5

lw=1.5
#dataset initialization
ds=MMParser(maps_arr)
dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2,drop_last=True)



#generator creation
netG=Generator_V(ngpu).to(device)
netG.apply(weights_init)


netD=Discriminator_S(ngpu).to(device)
netD.apply(weights_init)

stri="vgan_vbn_specnorm_FL128"

criterion = nn.BCELoss()
fixed_noise = torch.randn(image_size, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 0.9
fake_label = 0
d_loss=[]
g_loss=[]

iters=0

it_arr=[]
    # Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# schedulerD
# schedulerG = StepLR(optimizerG, step_size=1, gamma=3.5)

for epoch in range(num_epochs):
	# schedulerG.step()
	
	# print('Epoch:', epoch,'LR:', schedulerG.get_lr())
	# print("\n")
	for i,data in enumerate(dataloader,0):
		netD.zero_grad()

		real_cpu = (data.unsqueeze(dim=1).type(torch.FloatTensor)).to(device)
		b_size = real_cpu.size(0)
		label = torch.full((b_size,), real_label, device=device)
		output = netD(real_cpu).view(-1)

		errD_real = criterion(output, label)
		errD_real.backward()
		D_x = output.mean().item()

		noise = torch.randn(b_size, nz, 1, 1, device=device)
		fake = netG(noise)
		label.fill_(fake_label)
		sym_fake = (fake.clamp(min=0) + fake.clamp(min=0).permute(0, 1, 3, 2)) / 2
		output = netD(sym_fake.detach()).view(-1)
		errD_fake = criterion(output, label)
		errD_fake.backward()
		errD = errD_real + errD_fake
		optimizerD.step()
		# print(str(get_lr(optimizerD)))
		# print("   ")



		netG.zero_grad()
		label.fill_(real_label)  # fake labels are real for generator cost

		output = netD(sym_fake).view(-1)

		errG = criterion(output, label)
		errG.backward()
		# lrg=10e-4*(iters%1200)*5

		optimizerG.step()
		# print(str(get_lr(optimizerG)))
		# print("\n")

		g_loss.append(errG.item())
		d_loss.append(errD.item())
		it_arr.append(iters/773)
		iters+=1
		# break
	'''if(epoch==10 or epoch==20 or epoch==30 or epoch==50 or epoch==70 or epoch==100):
		torch.save(netG.state_dict(),"/scratch/trahman2/codes_128/"+stri+"_epochs"+str(epoch)+".pth")'''

# torch.save(netG.state_dict(),"/scratch/trahman2/h.pth")


plt.figure(figsize=(5.5,3.9),dpi=300)
# plt.title("Vanilla GAN")
plt.plot(it_arr,g_loss,label="G",color='b')
plt.plot(it_arr,d_loss,label="D",color='r')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.axvline(x=20,color='k',linestyle='--',linewidth=lw)
plt.axvline(x=30,color='k',linestyle='--',linewidth=lw)
plt.axvline(x=50,color='k',linestyle='--',linewidth=lw)
plt.axvline(x=70,color='k',linestyle='--',linewidth=lw)
plt.savefig("/scratch/trahman2/codes_128/"+stri+".png",bbox_inches="tight",pad_inches = 0)








