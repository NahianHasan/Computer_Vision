import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import scipy.ndimage as ndimage


building = mpimg.imread('./building.jpg')
nighthawks = mpimg.imread('./nighthawks.jpg')

'''
print(nighthawks.shape)
imgplot = plt.imshow(nighthawks)
plt.show()
'''

#Physical coordinates
building_l1 = np.array([[75,209],[180,61]])
building_l2 = np.array([[180,61],[744,283]])
building_l3 = np.array([[221,140],[239,125]])
building_l4 = np.array([[239,125],[263,90]])
building_l5 = np.array([[300,210],[717,324]])
building_l6 = np.array([[346,223],[344,380]])
building_l7 = np.array([[300,210],[717,324]])
building_l8 = np.array([[300,210],[297,376]])
building_l9 = np.array([[300,210],[717,324]])
building_l10 = np.array([[669,311],[671,406]])

nighthawks_l1 = np.array([[805,621],[805,589]])
nighthawks_l2 = np.array([[805,621],[773,621]])
nighthawks_l3 = np.array([[76,180],[76,212]])
nighthawks_l4 = np.array([[76,180],[180,113]])
nighthawks_l5 = np.array([[862,159],[865,305]])
nighthawks_l6 = np.array([[862,159],[632,142]])
nighthawks_l7 = np.array([[827,198],[827,304]])
nighthawks_l8 = np.array([[827,198],[634,186]])
nighthawks_l9 = np.array([[58,680],[231,672]])
nighthawks_l10 = np.array([[58,680],[56,566]])

def cross_product(A,B):
	result = np.array([A[1]*B[2]-A[2]*B[1],B[0]*A[2]-A[0]*B[2],A[0]*B[1]-B[0]*A[1]])
	return result

def transform_image(H,input_image,target_image):
	shape = np.array(target_image.shape)
	shape[0:2] = shape[0:2]/6
	transformed_image = np.zeros(shape)
	for i in range(0,input_image.shape[0]):
		for j in range(0,input_image.shape[1]):
			h_coordinate = np.array([i,j,1])
			t = np.matmul(H,h_coordinate)
			t = np.rint(t/t[2])
			t = t.astype('int')
			if (t[0]<transformed_image.shape[0] and t[1]<transformed_image.shape[1]):
				try:
					transformed_image[t[0],t[1],0] = input_image[i,j,0]
					transformed_image[t[0],t[1],1] = input_image[i,j,1]
					transformed_image[t[0],t[1],2] = input_image[i,j,2]
				except:
					continue
	return transformed_image

def form_homography_matrix(L1,L2,L3,L4,L5,L6,L7,L8,L9,L10):
	LM = np.zeros((10,3))
	LM[0,:] = cross_product(np.array([L1[0,0],L1[0,1],1]),np.array([L1[1,0],L1[1,1],1]))
	LM[1,:] = cross_product(np.array([L2[0,0],L2[0,1],1]),np.array([L2[1,0],L2[1,1],1]))
	LM[2,:] = cross_product(np.array([L3[0,0],L3[0,1],1]),np.array([L3[1,0],L3[1,1],1]))
	LM[3,:] = cross_product(np.array([L4[0,0],L4[0,1],1]),np.array([L4[1,0],L4[1,1],1]))
	LM[4,:] = cross_product(np.array([L5[0,0],L5[0,1],1]),np.array([L5[1,0],L5[1,1],1]))
	LM[5,:] = cross_product(np.array([L6[0,0],L6[0,1],1]),np.array([L6[1,0],L6[1,1],1]))
	LM[6,:] = cross_product(np.array([L7[0,0],L7[0,1],1]),np.array([L7[1,0],L7[1,1],1]))
	LM[7,:] = cross_product(np.array([L8[0,0],L8[0,1],1]),np.array([L8[1,0],L8[1,1],1]))
	LM[8,:] = cross_product(np.array([L9[0,0],L9[0,1],1]),np.array([L9[1,0],L9[1,1],1]))
	LM[9,:] = cross_product(np.array([L10[0,0],L10[0,1],1]),np.array([L10[1,0],L10[1,1],1]))

	for ix in range(0,10):
		LM[ix,:] = LM[ix,:]/LM[ix,-1]

	Coeffs = np.zeros((5,5))
	b = np.zeros((5,1))
	cnt=0
	for ix in range(0,10,2):
		Coeffs[cnt,:] = [LM[ix,0]*LM[ix+1,0],LM[ix,0]*LM[ix+1,1]+LM[ix,1]*LM[ix+1,0],LM[ix,1]*LM[ix+1,1],LM[ix,0]+LM[ix+1,0],LM[ix,1]+LM[ix+1,1]]
		b[cnt,0] = -1
		cnt = cnt + 1

	s = np.dot(np.linalg.inv(Coeffs),b)
	s = s/np.max(s)
	S = np.zeros((2,2))
	S[0,0] = s[0]
	S[0,1] = s[1]
	S[1,0] = s[1]
	S[1,1] = s[2]

	[V,lamb,VT] = np.linalg.svd(S)
	D = np.sqrt(np.diag(lamb))
	A = np.dot(np.dot(V,D),V.T)
	y=np.array([s[3],s[4]])
	y=np.reshape(y,[2,1])
	v = np.dot(np.linalg.inv(A),y)
	H = np.zeros((3,3))
	H[0:2,0:2] = A
	H[2,0:2] = v.T
	H[2,2] = 1
	print(H)
	return H


H = form_homography_matrix(building_l1,building_l2,building_l3,building_l4,building_l5,building_l6,building_l7,building_l8,building_l9,building_l10)
TI = transform_image(H,building,building)
transformed_image = TI.astype(np.uint8)
plt.imsave("building_transformed_1_step.png",transformed_image)

H = form_homography_matrix(nighthawks_l1,nighthawks_l2,nighthawks_l3,nighthawks_l4,nighthawks_l5,nighthawks_l6,nighthawks_l7,nighthawks_l8,nighthawks_l9,nighthawks_l10)
TI = transform_image(H,nighthawks,nighthawks)
transformed_image = TI.astype(np.uint8)
plt.imsave("nighthawks_transformed_1_step.png",transformed_image)
