import torch
import numpy as np
from utils import Bhattacharyya,short_range, long_range,score 
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance

def gen(stri):
	# stri="vgan_FL64_epochs50"
	fake=[]
	for i in range(0,774):
		tensor=torch.load("/scratch/trahman2/files_128/"+stri+str(i)+".txt")
		for elem in tensor:
			fake.append(elem.detach().numpy())
	return fake

def origin():
	f=open("/scratch/trahman2/train_idx.txt")
	fr=f.readlines()
	f.close()
	id_arr=[]
	for lines in fr:
		cont=lines.split(" ")
		if(int(cont[1])>=128):##make it only int(cont[1])==128 for 128x128 maps. 
			id_arr.append(cont[0])
	maps_arr=[]


	for id in id_arr:

		f_cont=np.loadtxt("/scratch/trahman2/map_128_1/"+id+".txt")
		maps_arr.append(f_cont)

	return maps_arr

models_arr=[]
epochs=[10]
fl="FL128"
map_size=128
th=8
# names=["vgan","vgan_vbn","vgan_specnorm","wgan"]
names=["vgan","vgan_vbn"]
for i in epochs:
	for j in names:
		model_name=j+"_"+fl+"_epochs"+str(i)
		models_arr.append(model_name)

real_arr=origin()
for model in models_arr:
	fake_arr=gen(model)
	gen_arr=score(fake_arr)
	real,fake=long_range(real_arr,fake_arr,map_size,th)
	real2,fake2=short_range(real_arr,fake_arr,map_size,th)
	if len(real_arr)>=len(fake_arr):
		a=len(fake_arr)
	else:
		a=len(real_arr)


	print("backbone stats for "+model)
	print("mean is")
	print(np.mean(gen_arr))
	print("median is")
	print(np.median(gen_arr))
	print("max is")
	print(np.amax(gen_arr))
	print("min is")
	print(np.amin(gen_arr))

	print("long range stats for " +model)
	print("BD for map_size "+str(map_size))
	print(Bhattacharyya(fake,real))
	print("EMD for map_size "+str(map_size))
	print(wasserstein_distance(fake,real))
	print("MMD for map_size "+str(map_size))
	print(np.linalg.norm(np.array(fake[:a])-np.array(real[:a])))

	print("short range stats for " +model)
	print("BD for map_size "+str(map_size))
	print(Bhattacharyya(fake2,real2))
	print("EMD for map_size "+str(map_size))
	print(wasserstein_distance(fake2,real2))
	print("MMD for map_size "+str(map_size))
	print(np.linalg.norm(np.array(fake2[:a])-np.array(real2[:a])))






