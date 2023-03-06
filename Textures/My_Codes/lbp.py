import os,math
import numpy as np
import cv2
import BitVector

def lbp(im,R=1,P=8,eps=0.000001,verbose=False):
	#calculates the local binary pattern of an image
	#im : input image
	#R : radius of the circular pattern
	#P : number of points to sample on the circle
	if len(im.shape)>2:
		#convert the image to grayscale
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	image_width = im.shape[1]
	image_height = im.shape[0]
	LBP_HIST = {kx:0 for kx in range(P+2)}

	for ix in range(R,image_height-R):
		for jx in range(R,image_width-R):
			#for each of these pixels generate a binary pattern
			PATTERN = list()
			for p in range(0,P):
				NEIGHBOUR_POINTS_ON_CIRCLE = np.array([R*np.cos(2*np.pi*p/P),R*np.sin(2*np.pi*p/P)])
				NEIGHBOUR_POINTS_ON_CIRCLE[abs(NEIGHBOUR_POINTS_ON_CIRCLE)<eps] = 0

				kx,lx = ix+NEIGHBOUR_POINTS_ON_CIRCLE[0],jx+NEIGHBOUR_POINTS_ON_CIRCLE[1]
				kxb,lxb = math.floor(kx),math.floor(lx)
				del_kx,del_lx = kx-kxb,lx-lxb
				if del_kx < eps and del_lx<eps:
					IMAGE_AT_P = im[kxb,lxb]
				elif del_lx < eps:
					IMAGE_AT_P = (1-del_kx)*im[kxb,lxb] + del_kx*im[kxb+1,lxb]
				elif del_kx < eps:
					IMAGE_AT_P = (1-del_lx)*im[kxb,lxb] + del_lx*im[kxb,lxb+1]
				else:
					IMAGE_AT_P = im[kxb,lxb]*(1-del_kx)*(del_lx)+im[kxb,lxb+1]*del_lx*(1-del_kx)+im[kxb+1,lxb]*del_kx*(1-del_lx)+im[kxb+1,lxb+1]*del_kx*del_lx;
				if IMAGE_AT_P>=im[ix,jx]:
					PATTERN.append(1)
				else:
					PATTERN.append(0)

			BITVECTOR = BitVector.BitVector(bitlist=PATTERN)
			CIRC_SHIFT_INTVL = [int(BITVECTOR<<1) for _ in range(P)]
			MINBV = BitVector.BitVector(intVal=min(CIRC_SHIFT_INTVL),size=P)
			BVRUNS = MINBV.runs()
			if len(BVRUNS) > 2:
				LBP_HIST[P+1] += 1
				ENCODING = P+1
			elif len(BVRUNS) == 1 and BVRUNS[0][0] == '1':
				LBP_HIST[P] += 1
				ENCODING = P
			elif len(BVRUNS) == 1 and BVRUNS[0][0] == '0':
				LBP_HIST[0] += 1
				ENCODING = 0
			else:
				LBP_HIST[len(BVRUNS[1])] += 1
				ENCODING = len(BVRUNS[1])
			if verbose:
				print('Pixel [i,j] = [',ix,',',jx,']')
				print('Pattern = ',PATTERN)
				print('minbv = ',MINBV)
				print('encoding = ',ENCODING)
				print('\n')
	if verbose:
		print('LBP Histogram = ',LBP_HIST)
	return LBP_HIST
