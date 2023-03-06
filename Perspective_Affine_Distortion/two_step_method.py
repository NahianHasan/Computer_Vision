import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import scipy.ndimage as ndimage


building = mpimg.imread('./building.jpg')
nighthawks = mpimg.imread('./nighthawks.jpg')


print(nighthawks.shape)
imgplot = plt.imshow(nighthawks)
plt.show()


#Physical coordinates
building_l1 = np.array([[238,194],[718,325]])
building_l2 = np.array([[220,377],[720,410]])
building_l3 = np.array([[683,314],[686,408]])
building_l4 = np.array([[709,323],[711,408]])

nighthawks_l1 = np.array([[77,654],[805,621]])
nighthawks_l2 = np.array([[14,729],[865,679]])
nighthawks_l3 = np.array([[12,100],[863,160]])
nighthawks_l4 = np.array([[56,151],[826,197]])


def cross_product(A,B):
	result = np.array([A[1]*B[2]-A[2]*B[1],B[0]*A[2]-A[0]*B[2],A[0]*B[1]-B[0]*A[1]])
	return result

def transform_image(H,input_image,target_image):
	shape = np.array(target_image.shape)
	shape[0:2] = shape[0:2]*2
	transformed_image = np.zeros(shape)
	for i in range(0,input_image.shape[0]):
		for j in range(0,input_image.shape[1]):
			h_coordinate = np.array([i,j,1])
			t = np.dot(H,h_coordinate)
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

def form_projective_homography_matrix(L1,L2,L3,L4):
	PL1 = cross_product(np.array([L1[0,0],L1[0,1],1]),np.array([L1[1,0],L1[1,1],1]))
	PL2 = cross_product(np.array([L2[0,0],L2[0,1],1]),np.array([L2[1,0],L2[1,1],1]))
	VP1 = cross_product(PL1,PL2)
	VP1 = VP1/VP1[-1]
	PL3 = cross_product(np.array([L3[0,0],L3[0,1],1]),np.array([L3[1,0],L3[1,1],1]))
	PL4 = cross_product(np.array([L4[0,0],L4[0,1],1]),np.array([L4[1,0],L4[1,1],1]))
	VP2 = cross_product(PL3,PL4)
	VP2 = VP2/VP2[-1]
	VL = cross_product(VP1,VP2)
	VL = VL/VL[-1]
	H_proj = np.zeros((3,3))
	H_proj[0,0] = 1;
	H_proj[1,1] = 1;
	H_proj[2,0] = VL[0];
	H_proj[2,1] = VL[1];
	H_proj[2,2] = VL[2];
	return H_proj

def form_affine_homography_matrix(L1,M1,L2,M2):
	PL1 = cross_product(np.array([L1[0,0],L1[0,1],1]),np.array([L1[1,0],L1[1,1],1]))
	PM1 = cross_product(np.array([M1[0,0],M1[0,1],1]),np.array([M1[1,0],M1[1,1],1]))
	PL2 = cross_product(np.array([L2[0,0],L2[0,1],1]),np.array([L2[1,0],L2[1,1],1]))
	PM2 = cross_product(np.array([M2[0,0],M2[0,1],1]),np.array([M2[1,0],M2[1,1],1]))
	#Solve the system of equations for S parameters
	Coeffs = np.array([[PL1[0]*PM1[0],PL1[0]*PM1[1]+PL1[1]*PM1[0]],[PL2[0]*PM2[0],PL2[0]*PM2[1]+PL2[1]*PM2[0]]])
	b = np.array([[-PL1[1]*PM1[1]],[-PL2[1]*PM2[1]]])
	b = np.reshape(b,(2,1))
	s = np.dot(np.linalg.inv(Coeffs),b)
	s = s/np.max(s)
	S = np.zeros((2,2))
	S[0,0] = s[0]
	S[0,1] = s[1]
	S[1,0] = s[1]
	S[1,1] = 1
	#print("S=",S)
	[V,lamb,VT] = np.linalg.svd(S)
	d = np.sqrt(lamb)
	D = np.zeros((2,2))
	D[0,0] = d[0]
	D[1,1] = d[1]
	A = np.dot(np.dot(V,D),V.T)
	#print("A = ",A)
	H_affine = np.zeros((3,3))
	H_affine[0:2,0:2] = A
	H_affine[2,2] = 1
	return H_affine

plt.figure()
imgplot = plt.imshow(building)
plt.plot([building_l1[0,0],building_l1[1,0]], [building_l1[0,1],building_l1[1,1]],'-b',linewidth=3)
plt.plot([building_l2[0,0],building_l2[1,0]], [building_l2[0,1],building_l2[1,1]],'-b',linewidth=3)
plt.plot([building_l3[0,0],building_l3[1,0]], [building_l3[0,1],building_l3[1,1]],'-r',linewidth=3)
plt.plot([building_l4[0,0],building_l4[1,0]], [building_l4[0,1],building_l4[1,1]],'-r',linewidth=3)
plt.savefig("building_transformed_2_step_parallel_lines.png")
###   Step 1 ####
H_proj = form_projective_homography_matrix(building_l1,building_l2,building_l3,building_l4)
print(H_proj)
TI = transform_image(H_proj,building,building)
transformed_image = TI.astype(np.uint8)
plt.imsave("building_transformed_2_step_1.png",transformed_image)
###   Step 2 ####
building_l1 = np.array([[75,209],[180,61]])
building_m1 = np.array([[180,61],[744,283]])
building_l2 = np.array([[221,140],[239,125]])
building_m2 = np.array([[239,125],[277,134]])
plt.figure()
imgplot = plt.imshow(building)
plt.plot([building_l1[0,0],building_l1[1,0]], [building_l1[0,1],building_l1[1,1]],'-b',linewidth=3)
plt.plot([building_m1[0,0],building_m1[1,0]], [building_m1[0,1],building_m1[1,1]],'-b',linewidth=3)
plt.plot([building_l2[0,0],building_l2[1,0]], [building_l2[0,1],building_l2[1,1]],'-r',linewidth=3)
plt.plot([building_m2[0,0],building_m2[1,0]], [building_m2[0,1],building_m2[1,1]],'-r',linewidth=3)
plt.savefig("building_transformed_2_step_perpendicular_lines.png")
H_affine = form_affine_homography_matrix(building_l1,building_m1,building_l2,building_m2)
print(H_affine)
transformed_image = transform_image(np.dot(H_affine,H_proj),building,building)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("building_transformed_2_step_2.png",transformed_image)


###   Step 1 ####
plt.figure()
imgplot = plt.imshow(nighthawks)
plt.plot([nighthawks_l1[0,0],nighthawks_l1[1,0]], [nighthawks_l1[0,1],nighthawks_l1[1,1]],'-b',linewidth=3)
plt.plot([nighthawks_l2[0,0],nighthawks_l2[1,0]], [nighthawks_l2[0,1],nighthawks_l2[1,1]],'-b',linewidth=3)
plt.plot([nighthawks_l3[0,0],nighthawks_l3[1,0]], [nighthawks_l3[0,1],nighthawks_l3[1,1]],'-r',linewidth=3)
plt.plot([nighthawks_l4[0,0],nighthawks_l4[1,0]], [nighthawks_l4[0,1],nighthawks_l4[1,1]],'-r',linewidth=3)
plt.savefig("nighthawks_transformed_2_step_parallel_lines.png")
H_proj = form_projective_homography_matrix(nighthawks_l1,nighthawks_l2,nighthawks_l3,nighthawks_l4)
print(H_proj)
TI = transform_image(H_proj,nighthawks,nighthawks)
transformed_image = TI.astype(np.uint8)
plt.imsave("nighthawks_transformed_2_step_1.png",transformed_image)
###   Step 2 ####
nighthawks_l1 = np.array([[805,621],[805,589]])
nighthawks_m1 = np.array([[805,621],[773,621]])
nighthawks_l2 = np.array([[76,180],[75,252]])
nighthawks_m2 = np.array([[76,180],[190,184]])
plt.figure()
imgplot = plt.imshow(nighthawks)
plt.plot([nighthawks_l1[0,0],nighthawks_l1[1,0]], [nighthawks_l1[0,1],nighthawks_l1[1,1]],'-b',linewidth=3)
plt.plot([nighthawks_m1[0,0],nighthawks_m1[1,0]], [nighthawks_m1[0,1],nighthawks_m1[1,1]],'-b',linewidth=3)
plt.plot([nighthawks_l2[0,0],nighthawks_l2[1,0]], [nighthawks_l2[0,1],nighthawks_l2[1,1]],'-r',linewidth=3)
plt.plot([nighthawks_m2[0,0],nighthawks_m2[1,0]], [nighthawks_m2[0,1],nighthawks_m2[1,1]],'-r',linewidth=3)
plt.savefig("nighthawks_transformed_2_step_perpendicular_lines.png")
H_affine = form_affine_homography_matrix(nighthawks_l1,nighthawks_m1,nighthawks_l2,nighthawks_m2)
print(H_affine)
transformed_image = transform_image(np.dot(H_affine,H_proj),nighthawks,nighthawks)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("nighthawks_transformed_2_step_2.png",transformed_image)
