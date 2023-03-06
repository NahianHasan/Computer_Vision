import os
import numpy as np


def haar_kernel(sigma):
	N = int(np.ceil(4*sigma))
	if N%2 == 1:
		N = N+1
	h_x = np.ones((N,N))
	h_y = np.ones((N,N))
	h_x[:,0:int(N/2)]=-1
	h_y[int(N/2):-1,:] = -1
	return h_x,h_y,N

def convolution(image,filter,padding_mode='constant'):
	[height,width] = image.shape
	N = filter.shape[0]
	result = np.zeros((height,width))
	#Pad the image for handling corner operations
	image = np.pad(image,((int(N/2),int(N/2)-1),(int(N/2),int(N/2)-1)),mode=padding_mode)
	for i in range(int(N/2),image.shape[0]-(int(N/2)-1)):
		for j in range(int(N/2),image.shape[1]-(int(N/2)-1)):
			im_neighbour = image[i-int(N/2):i+int(N/2),j-int(N/2):j+int(N/2)]
			result[i-int(N/2),j-int(N/2)] = np.sum(np.multiply(im_neighbour,filter))
	#print(np.unique(result.flatten()))
	return result

def harris_detector(Im,sigma,Th):
	#Im = input image
	#sigma  = Scale factor
	#Th = Threshold percentile
	#Outputs:
	#corners : binary image of same shape as I with 1=corners, 0 = not corners
	#ratios : determinant(C)/Trace(C)^2, can be used later for thresholding with a separate threshold
	#define kernels

	Im = np.array(Im)
	if len(Im.shape)>2:
		Im = Im[:,:,0]
	Im = Im/255

	[h_x,h_y,N] = haar_kernel(sigma)
	dx = convolution(Im,h_x)
	dy = convolution(Im,h_y)
	dx2 = np.multiply(dx,dx)
	dy2 = np.multiply(dy,dy)
	dxdy = np.multiply(dx,dy)

	N_sum = int(np.ceil(5*sigma))
	if N_sum%2 == 1:
		N_sum = N_sum+1
	sum_filter = np.ones((N_sum,N_sum))
	dx2 = convolution(dx2,sum_filter)
	dy2 = convolution(dy2,sum_filter)
	dxdy = convolution(dxdy,sum_filter)
	Tr_C = dx2 + dy2
	det_C = np.multiply(dx2,dy2) - np.multiply(dxdy,dxdy)

	k_tmp = det_C/np.square(Tr_C)
	k_tmp[np.isnan(k_tmp)] = 0
	k = np.sum(k_tmp)/(Im.shape[0]*Im.shape[1])
	R = det_C - k*np.square(Tr_C)
	Th = np.percentile(R, Th)
	print('R_threshold = ',Th)
	corner_indexes = np.argwhere(R>=Th)
	print('Number of corners = ',corner_indexes.shape[0])

	return corner_indexes,Th
