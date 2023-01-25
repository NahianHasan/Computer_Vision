import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import scipy.ndimage as ndimage


card1 = mpimg.imread('./card1.jpeg')
card2 = mpimg.imread('./card2.jpeg')
card3 = mpimg.imread('./card3.jpeg')
car = mpimg.imread('./car.jpg')

'''
print(car.shape)
imgplot = plt.imshow(car)
# Create a Rectangle patch
rect = patches.Rectangle((21, 31), 700, 500, linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
ax = plt.gca();
ax.add_patch(rect)
plt.savefig('car_bounding_box.png')
#plt.show()
'''

#Physical coordinates
card1_coord = np.array([[488,252],[610,1114],[1220,800],[1241,177]])
card2_coord = np.array([[318,230],[218,864],[872,1128],[1042,232]])
card3_coord = np.array([[586,47],[61,591],[701,1214],[1230,675]])
car_coord = np.array([[21,31],[21,531],[721,531],[721,31]])

def homography_coefficients(input_coord,target_coord,mode=None):
	A = np.array([[input_coord[0,0],input_coord[0,1],1,0,0,0,-input_coord[0,0]*target_coord[0,0],-input_coord[0,1]*target_coord[0,0]],\
	[0,0,0,input_coord[0,0],input_coord[0,1],1,-input_coord[0,0]*target_coord[0,1],-input_coord[0,1]*target_coord[0,1]],\
	[input_coord[1,0],input_coord[1,1],1,0,0,0,-input_coord[1,0]*target_coord[1,0],-input_coord[1,1]*target_coord[1,0]],\
	[0,0,0,input_coord[1,0],input_coord[1,1],1,-input_coord[1,0]*target_coord[1,1],-input_coord[1,1]*target_coord[1,1]],\
	[input_coord[2,0],input_coord[2,1],1,0,0,0,-input_coord[2,0]*target_coord[2,0],-input_coord[2,1]*target_coord[2,0]],\
	[0,0,0,input_coord[2,0],input_coord[2,1],1,-input_coord[2,0]*target_coord[2,1],-input_coord[2,1]*target_coord[2,1]],\
	[input_coord[3,0],input_coord[3,1],1,0,0,0,-input_coord[3,0]*target_coord[3,0],-input_coord[3,1]*target_coord[3,0]],\
	[0,0,0,input_coord[3,0],input_coord[3,1],1,-input_coord[3,0]*target_coord[3,1],-input_coord[3,1]*target_coord[3,1]]])

	b = np.array([[target_coord[0,0]],[target_coord[0,1]],[target_coord[1,0]],[target_coord[1,1]],[target_coord[2,0]],[target_coord[2,1]],[target_coord[3,0]],[target_coord[3,1]]])
	x = np.matmul(np.linalg.inv(A),b)

	H = np.zeros((3,3))
	H[0,0] = x[0]
	H[0,1] = x[1]
	H[0,2] = x[2]
	H[1,0] = x[3]
	H[1,1] = x[4]
	H[1,2] = x[5]
	H[2,0] = x[6]
	H[2,1] = x[7]
	H[2,2] = 1
	if mode=='affine':
		H[2,0] = 0
		H[2,1] = 0
	return H

def transform_image(H,input_image,target_image):
	transformed_image = np.zeros(target_image.shape)
	for i in range(0,input_image.shape[0]):
		for j in range(0,input_image.shape[1]):
			h_coordinate = np.array([j,i,1])
			t = np.matmul(H,h_coordinate)
			t = np.rint(t/t[2])
			t = t.astype('int')
			if (t[0]<target_image.shape[0] and t[1]<target_image.shape[1]):
				transformed_image[t[1],t[0],0] = input_image[i,j,0]
				transformed_image[t[1],t[0],1] = input_image[i,j,1]
				transformed_image[t[1],t[0],2] = input_image[i,j,2]
	'''
	for i in range(1,input_image.shape[0]-1):
		for j in range(1,input_image.shape[1]-1):
			sum0 = 0
			sum1=0
			sum2=0
			for k in range(i-1,i+2):
				for l in range(j-1,j+2):
					sum0 = sum0 + transformed_image[k,l,0]
					sum1 = sum1 + transformed_image[k,l,1]
					sum2 = sum2 + transformed_image[k,l,2]

			transformed_image[i,j,0] = sum0/9
			transformed_image[i,j,1] = sum1/9
			transformed_image[i,j,2] = sum2/9
	'''
	return transformed_image

#########################   Task 1 #############################################
H = homography_coefficients(car_coord,card1_coord)
transformed_image = transform_image(H,car,card1)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("car_transformed_to_card1.png",transformed_image)

H = homography_coefficients(car_coord,card2_coord)
transformed_image = transform_image(H,car,card2)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("car_transformed_to_card2.png",transformed_image)

H = homography_coefficients(car_coord,card3_coord)
transformed_image = transform_image(H,car,card3)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("car_transformed_to_card3.png",transformed_image)

H12 = homography_coefficients(card1_coord,card2_coord)
H23 = homography_coefficients(card2_coord,card3_coord)
transformed_image = transform_image(np.matmul(H23,H12),card1,card3)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("card1_transformed_to_card3.png",transformed_image)

H = homography_coefficients(car_coord,card1_coord,mode='affine')
transformed_image = transform_image(H,car,card1)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("car_transformed_to_card1_affine.png",transformed_image)

H = homography_coefficients(car_coord,card2_coord,mode='affine')
transformed_image = transform_image(H,car,card2)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("car_transformed_to_card2_affine.png",transformed_image)

H = homography_coefficients(car_coord,card3_coord,mode='affine')
transformed_image = transform_image(H,car,card3)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("car_transformed_to_card3_affine.png",transformed_image)

#########################   Task 2 #############################################
custom_image_1 = mpimg.imread('./custom_image_1.jpg')
custom_image_2 = mpimg.imread('./custom_image_2.jpg')
custom_image_3 = mpimg.imread('./custom_image_3.jpg')
custom_projecting_image = mpimg.imread('./custom_projecting_image.jpg')
#Physical coordinates
custom_image_1_coord = np.array([[658,353],[31,1230],[2932,1352],[2717,320]])
custom_image_2_coord = np.array([[831,332],[49,1216],[2618,1726],[2873,445]])
custom_image_3_coord = np.array([[278,660],[341,2146],[2704,2476],[2719,250]])
custom_projecting_image_coord = np.array([[235,424],[140,2025],[2736,2186],[2722,409]])
#plt.imshow(custom_projecting_image)
#plt.show()

H = homography_coefficients(custom_projecting_image_coord,custom_image_1_coord)
transformed_image = transform_image(H,custom_projecting_image,custom_image_1)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("cst_transformed_to_cst1.png",transformed_image)

H = homography_coefficients(custom_projecting_image_coord,custom_image_2_coord)
transformed_image = transform_image(H,custom_projecting_image,custom_image_2)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("cst_transformed_to_cst2.png",transformed_image)

H = homography_coefficients(custom_projecting_image_coord,custom_image_3_coord)
transformed_image = transform_image(H,custom_projecting_image,custom_image_3)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("cst_transformed_to_cst3.png",transformed_image)

H12 = homography_coefficients(custom_image_1_coord,custom_image_2_coord)
H23 = homography_coefficients(custom_image_2_coord,custom_image_3_coord)
transformed_image = transform_image(np.matmul(H23,H12),custom_image_1,custom_image_3)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("cst1_transformed_to_cst3.png",transformed_image)

H = homography_coefficients(custom_projecting_image_coord,custom_image_1_coord,mode='affine')
transformed_image = transform_image(H,custom_projecting_image,custom_image_1)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("cst_transformed_to_cst1_affine.png",transformed_image)

H = homography_coefficients(custom_projecting_image_coord,custom_image_2_coord,mode='affine')
transformed_image = transform_image(H,custom_projecting_image,custom_image_2)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("cst_transformed_to_cst2_affine.png",transformed_image)

H = homography_coefficients(custom_projecting_image_coord,custom_image_3_coord,mode='affine')
transformed_image = transform_image(H,custom_projecting_image,custom_image_3)
transformed_image = transformed_image.astype(np.uint8)
plt.imsave("cst_transformed_to_cst3_affine.png",transformed_image)
