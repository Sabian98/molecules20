
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
from torch.autograd import Variable
from torchgan.layers import VirtualBatchNorm
#
from classes import MMParser, Generator_V,Generator, Discriminator_S, Discriminator
scaled_factor=100


lw=1.5
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
# for i  in range(1000):
# 	id=id_arr[i]#1000 for initial testing

	
	f_cont=np.loadtxt("/scratch/trahman2/map_128_1/"+id+".txt")
	maps_arr.append(f_cont)
# print(maps_arr[0])
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

stri="wgan_FL128"


#dataset initialization
ds=MMParser(maps_arr)
dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2,drop_last=True)



#generator creation
netG=Generator(ngpu).to(device)
# netG.apply(weights_init)


netD=Discriminator(ngpu).to(device)
# netD.apply(weights_init)

# criterion = nn.BCELoss()
fixed_noise = torch.randn(image_size, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 0.9
fake_label = 0
d_loss=[]
g_loss=[]
it_arr=[]

iters=0
    # Setup Adam optimizers for both G and D
optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

# schedulerD
# schedulerG = StepLR(optimizerG, step_size=1, gamma=3.5)

for epoch in range(num_epochs):
	# schedulerG.step()
	
	# print('Epoch:', epoch,'LR:', schedulerG.get_lr())
	# print("\n")
	for i,data in enumerate(dataloader,0):
		netD.zero_grad()

		real_imgs = (data.unsqueeze(dim=1).type(torch.FloatTensor)).to(device)

		b_size = real_imgs.size(0)

		noise = torch.randn(b_size, nz, 1, 1, device=device)
		
		fake_imgs=netG(noise).detach()

		# Adversarial loss
		loss_D = -torch.mean(netD(real_imgs)) + torch.mean(netD(fake_imgs))

		loss_D.backward()
		optimizerD.step()
		

		# Clip weights of discriminator
		for p in netD.parameters():
			p.data.clamp_(-.01,.01)

		
		optimizerG.zero_grad()
		gen_imgs = netG(noise)
		loss_G = -torch.mean(netD(gen_imgs))

		loss_G.backward()
		optimizerG.step()
		d_loss.append(loss_D.item())
		g_loss.append(loss_G.item())
		it_arr.append(iters/880)
		
		iters+=1
	if(epoch==10 or epoch==20 or epoch==30 or epoch==50 or epoch==70 or epoch==100):
		torch.save(netG.state_dict(),"/scratch/trahman2/codes_128/"+stri+"_epochs"+str(epoch)+".pth")
		'''if( epoch==50):
			
			with torch.no_grad():
				fake = netG(fixed_noise).detach().cpu()
				fake = (fake.clamp(min=0) + fake.clamp(min=0).permute(0, 1, 3, 2)) / 2
				fake=torch.squeeze(fake)
		# 	print(fake.size())
				torch.save(fake,"/scratch/trahman2/final2/wgan_6/map_"+str(iters%880)+".pt")'''
	
	

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


















# 	print(count)
# print(len(maps_arr))	
