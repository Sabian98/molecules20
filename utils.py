import torch
import time
import numpy as np

import matplotlib.pyplot as plt
import random



def Bhattacharyya(x,y):

	plt.figure(figsize=(8,4), dpi=80) 
	cnt_x = plt.hist(x, bins=20,width=0.5)
	cnt_y = plt.hist(y, bins=20)
	x_=cnt_x[0]/len(x)   # No. of points in bin divided by total no. of samples.
	y_=cnt_y[0]/len(y)    
	BC=np.sum(np.sqrt(x_*y_))
	plt.close()
	return -np.log(BC)

def short_range(real_arr,fake_arr,map_size,th):
	fake_dist1=[]
	for f_cont in fake_arr:
		count1=0
		for index in range(map_size):
			arr2=f_cont[index][index+2:index+5]
			a=arr2
			ref_tens2=torch.full((len(arr2),1),th,dtype=torch.long)
			b=ref_tens2.detach().numpy().squeeze()
			count1+=np.sum(b>a)
		fake_dist1.append(count1)


	real_dist=[]
	for f_cont in real_arr:
		count2=0
		for index in range(map_size):
			arr2=f_cont[index][index+2:index+5]
			# a=arr1.detach().numpy()
			ref_tens2=torch.full((len(arr2),1),th,dtype=torch.long)
			a=arr2
			b=ref_tens2.detach().numpy().squeeze()
			count2+=np.sum(b>a)
		real_dist.append(count2)

	return real_dist,fake_dist1

def long_range(real_arr,fake_arr,map_size,th):
	fake_dist1=[]
	for f_cont in fake_arr:
		count1=0
		for index in range(map_size):
			arr2=f_cont[index][index+2:]
			a=arr2
			ref_tens2=torch.full((len(arr2),1),th,dtype=torch.long)
			b=ref_tens2.detach().numpy().squeeze()
			count1+=np.sum(b>a)
		fake_dist1.append(count1)


	real_dist=[]
	for f_cont in real_arr:
		count2=0
		for index in range(map_size):
			arr2=f_cont[index][index+2:]
			# a=arr1.detach().numpy()
			ref_tens2=torch.full((len(arr2),1),th,dtype=torch.long)
			a=arr2
			b=ref_tens2.detach().numpy().squeeze()
			count2+=np.sum(b>a)
		real_dist.append(count2)

	return real_dist,fake_dist1

def get_score(res):
	count=0
	for i in range(len(res)-1):
		if(res[i][i+1]<=4):
			count+=1
	return count

def score(arr):

	gen_arr=[]
	for res in arr:
		tr=np.transpose(res)
		res=(res +tr)/2
		score=get_score(res)
		gen_arr.append(score)
	gen_arr=np.array(gen_arr)
	return gen_arr










