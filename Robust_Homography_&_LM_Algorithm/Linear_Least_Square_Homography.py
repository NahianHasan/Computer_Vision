import cv2
import numpy as np

def LLSM(points_1,points_2):
	A = np.zeros((2*points_1.shape[0],8))
	b = np.zeros((2*points_1.shape[0],1))
	for ix in range(0,points_1.shape[0]):
		A[2*ix,:] = np.array([points_1[ix,0],points_1[ix,1],1,0,0,0,-points_1[ix,0]*points_2[ix,0],-points_1[ix,1]*points_2[ix,0]])
		A[2*ix+1,:] = np.array([0,0,0,points_1[ix,0],points_1[ix,1],1,-points_1[ix,0]*points_2[ix,1],-points_1[ix,1]*points_2[ix,1]])
		b[2*ix] = points_2[ix,0]
		b[2*ix+1] = points_2[ix,1]

	H = np.matmul(calculate_pseudo_inverse(A),b)
	H = np.reshape(np.append(H,1),(3,3))
	return H

def calculate_pseudo_inverse(A):
	return np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T)
