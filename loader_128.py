
import numpy as np
import random
import os
import torch
from utils import Bhattacharyya,short_range, long_range,score 
from classes import MMParser, Generator_V,Generator, Discriminator_S, Discriminator
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance

#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu=0
if torch.cuda.is_available():
		ngpu = 2
	

manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Size of z latent vector (i.e. size of generator input)
nz = 100
fl="FL128"
map_size=128
image_size=128
th=8
files=98966###98966 for FL 128

def gen(stri):
	if 'vbn' in stri:
		netG=Generator_V(ngpu).to(device)
	else:
		netG=Generator(ngpu).to(device)
#here-'vgan' is the type of gan 
#there can be total 10 kinds of GAN..see the /scratch/trahman2/codes_FL for details
	netG.load_state_dict(torch.load("/scratch/trahman2/codes_128/"+stri+".pth",map_location=torch.device('cpu')))
	fake_arr=[]
	iter=0
	for _ in range(int(files/image_size)+1):

		fixed_noise = torch.randn(image_size, nz, 1, 1, device=device)

		fake=netG(fixed_noise).squeeze()
		torch.save(fake,"/scratch/trahman2/files_128/"+stri+str(iter)+".txt")
		iter+=1
		# for elem in fake:
		# 	iter+=1
		# 	np.savetxt("/scratch/trahman2/np_64/"+stri+str(iter)+".txt",elem.detach().numpy())
	print(iter)

	return fake_arr
#each fake tensor is of size image_size x FL_size x FL_size
#so we generate enough "fake" tensors that will be equal to the length original dataset 


#the following code reads and appends original maps to a list




models_arr=[]
epochs=[30]
names=["vgan","vgan_vbn","vgan_specnorm","wgan"]
for i in epochs:
	for j in names:
		model_name=j+"_"+fl+"_epochs"+str(i)
		models_arr.append(model_name)
# models_arr=["vgan_"+fl+"_epochs50","vgan_vbn_"+fl+"_epochs50","vgan_specnorm_"+fl+"_epochs50","wgan_"+fl+"_epochs50","wgan_ttur_"+fl+"_epochs50"]
# real_arr=origin()

for model in models_arr:
	fake_arr=gen(model)










