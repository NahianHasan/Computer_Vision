import cv2, os
import numpy as np
import math
from scipy.stats import entropy

def image_read(fname, gray_scale=False):
	if gray_scale:
		im = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
	else:
		im = cv2.imread(fname)
	return im
def image_texture(image,N,feature,out_dir,name):
	#N = window size = N*N
	save_dir = out_dir+'/'+name
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	gray_image = image
	if len(list(gray_image.shape))>2:
		gray_image = cv2.cvtColor(gray_image,cv2.COLOR_BGR2GRAY)
	textures = np.zeros((image.shape[0],image.shape[1],len(N)))

	for tx in range(0,len(N)):
		padding = int(N[tx]/2)
		padded_image = np.pad(gray_image,((padding,padding),(padding,padding)),mode='constant',constant_values=0)
		if feature.upper() == 'VAR':
			for ix in range(padding,padded_image.shape[0]-2*padding):
				for jx in range(padding,padded_image.shape[1]-2*padding):
					textures[ix,jx,tx] = np.var(padded_image[ix-padding:ix+padding+1,jx-padding:jx+padding+1])
		elif feature.upper() == "ENTROPY":
			for ix in range(padding,padded_image.shape[0]-2*padding):
				for jx in range(padding,padded_image.shape[1]-2*padding):
					val,cnt = np.unique(padded_image[ix-padding:ix+padding+1,jx-padding:jx+padding+1], return_counts=True)
					norm_cnt = cnt / cnt.sum()
					textures[ix,jx,tx] = -(norm_cnt * np.log(norm_cnt)/np.log(math.e)).sum()
		else:
			print("Please provide a valid feature name")
	cv2.imwrite(save_dir+"/textures_"+feature+"_"+name+".jpg",textures.astype(np.uint8))
	return textures
def otsu_gray_scale(epochs,image,out_dir,name):
	save_dir = out_dir+'/'+name.split("_")[0]
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	N = np.prod(image.shape)#Total number of pixels in the image
	mask = np.ones(image.shape)
	for ix in range(0,epochs):
		image_new = np.multiply(image,mask)
		sigma_optimum = 0
		k_optimum = 0
		hist,bin_edges = np.histogram(image_new,bins=np.arange(256))
		p = hist/N#probability of each bin
		for k in range(0,256):
			#k = grayscale threshold level
			w0 = np.sum(p[:k])
			w1 = np.sum(p[k:])
			P = np.multiply(p,np.arange(1,256,1))
			if w0>0 and w1>0:
				mu0 = np.sum(P[:k])/w0
				mu1 = np.sum(P[k:])/w1
				sigma = w0*w1*(mu0-mu1)**2
				if sigma>=sigma_optimum:
					sigma_optimum=sigma
					k_optimum=k
		mask = np.where(image_new<k_optimum,0,1)
		image = image_new
		print("Epoch = ",str(ix),' k_optimum = ',str(k_optimum),' sigma_optimum = ',str(sigma_optimum))
	cv2.imwrite(save_dir+'/'+name+'_otsu_gray_scale.jpg',(mask*255).astype(np.uint8))
	return mask,sigma_optimum,k_optimum
def otsu_RGB(epochs,image,out_dir,name,special_case=False):
	save_dir = out_dir+'/'+name.split("_")[0]
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	nb_channels = image.shape[-1]
	final_mask = []
	for ch in range(0,nb_channels):
		print('\n\nChannel = ',ch)
		mask,_,_ = otsu_gray_scale(epochs,image[:,:,ch],out_dir,name+'_ch_'+str(ch))
		final_mask.append(mask)
	if special_case:
		if name in ['cat','fox']:
			final_mask = np.logical_and(final_mask[2],final_mask[2]-np.logical_and(final_mask[0],final_mask[1]))
		elif name in ['car']:
			final_mask = np.logical_and(final_mask[1],final_mask[1]-np.logical_and(final_mask[2],final_mask[0]))
	else:
		final_mask = np.logical_and(final_mask[0],final_mask[1],final_mask[2])
	cv2.imwrite(save_dir+'/'+name+'_otsu_RGB.jpg',(final_mask*255).astype(np.uint8))
	return final_mask
def find_contour(mask,out_dir,name,reverse_contour=False):
	save_dir = out_dir+'/'+name.split("_")[0]
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	contour = np.zeros(mask.shape)
	padding=1
	mask = np.pad(mask,((padding,padding),(padding,padding)),mode='constant',constant_values=0)
	for ix in range(padding,mask.shape[0]-2*padding):
		for jx in range(padding,mask.shape[1]-2*padding):
			if mask[ix,jx] == 1:
				if 0 in mask[ix-padding:ix+padding+1,jx-padding:jx+padding+1]:
					contour[ix,jx] = 1

	if reverse_contour:
		contour = np.abs(-1+contour)
	cv2.imwrite(save_dir+'/'+name+'_otsu_RGB.jpg',(contour*255).astype(np.uint8))
	return contour

def Main():
	out_dir = './Output'

	image_name = "cat"
	fname = "../Images/"+image_name+".jpg"
	I1 = image_read(fname)
	mask_from_raw = otsu_RGB(10,I1,out_dir,image_name,special_case=True)
	contour = find_contour(mask_from_raw,out_dir,image_name+'_contour_raw')
	features = 'var'#options = "var","entropy"
	N = [3,5,7]
	textures = image_texture(I1,N,features,out_dir,image_name)
	mask_from_texture = otsu_RGB(10,textures,out_dir,image_name+'_texture')
	contour = find_contour(mask_from_texture,out_dir,image_name+'_contour_texture')


	image_name = "car"
	fname = "../Images/"+image_name+".jpg"
	I1 = image_read(fname)
	mask_from_raw = otsu_RGB(10,I1,out_dir,image_name,special_case=True)
	contour = find_contour(mask_from_raw,out_dir,image_name+'_contour_raw')
	features = 'var'#options = "var","entropy"
	N = [3,4,5]
	textures = image_texture(I1,N,features,out_dir,image_name)
	mask_from_texture = otsu_RGB(10,textures,out_dir,image_name+'_texture')
	contour = find_contour(mask_from_texture,out_dir,image_name+'_contour_texture')


	image_name = "fox"
	fname = "../Images/"+image_name+".jpg"
	I1 = image_read(fname)
	mask_from_raw = otsu_RGB(10,I1,out_dir,image_name,special_case=True)
	contour = find_contour(mask_from_raw,out_dir,image_name+'_contour_raw')
	features = 'var'#options = "var","entropy"
	N = [3,5,7]
	textures = image_texture(I1,N,features,out_dir,image_name)
	mask_from_texture = otsu_RGB(10,textures,out_dir,image_name+'_texture')
	contour = find_contour(mask_from_texture,out_dir,image_name+'_contour_texture')


	image_name = "portrait"
	fname = "../Images/"+image_name+".jpg"
	I1 = image_read(fname)
	mask_from_raw = otsu_RGB(10,I1,out_dir,image_name)
	contour = find_contour(mask_from_raw,out_dir,image_name+'_contour_raw')
	features = 'var'#options = "var","entropy"
	N = [3,5,7]
	textures = image_texture(I1,N,features,out_dir,image_name)
	mask_from_texture = otsu_RGB(10,textures,out_dir,image_name+'_texture')
	contour = find_contour(mask_from_texture,out_dir,image_name+'_contour_texture')
	'''
Main()
