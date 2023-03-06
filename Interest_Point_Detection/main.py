import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import harris_corner_detector as HCD
import correspondence_metrics as CM
import time
import cv2


def opencv_interest_points(im_path,name):
	img = cv2.imread(im_path)
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT_create()
	kp,des = sift.detectAndCompute(gray,None)
	img=cv2.drawKeypoints(gray,kp,img)
	cv2.imwrite('sift_keypoints_'+name+'.jpg',img)
	return img,kp,des

def opencv_correspondence(img_1,kp_1,des_1,img_2,kp_2,des_2,name,N):
	#N 	: how many correspondences to be kept
	bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)# create BFMatcher object
	matches = bf.match(des_1,des_2)# Match descriptors.
	matches = sorted(matches, key = lambda x:x.distance)# Sort them in the order of their distance.
	img3 = cv2.drawMatches(img_1,kp_1,img_2,kp_2,matches[:N], None,flags=2)# Draw first 10 matches.
	cv2.imwrite('opencv_correspondences_'+name+'.jpg',img3)

def plot_interest_points(Im,indexes,name,scale):
	for ix in range(0,indexes.shape[0]):
		cv2.circle(Im,indexes[ix,:],5,(255, 255, 0),1)
	cv2.imwrite('Interest_Points_'+name+'_'+str(scale)+'.jpg',Im)

def corner_detection_image_pairs(Im_1_path,Im_2_path,name_1,name_2,Th,scale):
	Im_1 = mpimg.imread(Im_1_path)
	Im_2 = mpimg.imread(Im_2_path)
	print("Image Files Read")

	start_time = time.time()
	[corner_indexes_1,Th_1] = HCD.harris_detector(Im_1,scale,Th)
	print("Interest Points Detected for "+name_1)
	end_time = time.time()
	[corner_indexes_2,Th_2] = HCD.harris_detector(Im_2,scale,Th)
	print("Interest Points Detected for "+name_2)
	print('Required Time for Harris Corner Detection Algorithm = ',str(end_time-start_time))
	plot_interest_points(Im_1,corner_indexes_1,name_1,scale)
	plot_interest_points(Im_2,corner_indexes_2,name_2,scale)
	print("Interest Points Plotted")
	correspondences_SSD = CM.Sum_of_Squared_Differences(Im_1,Im_2,corner_indexes_1,corner_indexes_2,21)
	correspondences_NCC = CM.Normalized_Cross_Correlations(Im_1,Im_2,corner_indexes_1,corner_indexes_2,21)
	CM.plot_correspondences(Im_1,Im_2,correspondences_SSD,corner_indexes_1,corner_indexes_2,10000,'SSD_'+name_1[:-2]+'_'+str(scale))
	CM.plot_correspondences(Im_1,Im_2,correspondences_NCC,corner_indexes_1,corner_indexes_2,10000,'NCC_'+name_1[:-2]+'_'+str(scale))


def main():
	scales=[0.8,1.2,1.6,2.0]

	im_1 = './HW4-Images/Figures/books_1.jpeg'
	im_2 = './HW4-Images/Figures/books_2.jpeg'
	for scale in scales:
		corner_detection_image_pairs(im_1,im_2,'books_1','books_2',99.99,scale)
		if scale==1.2:
			[img_1,kp_1,des_1] = opencv_interest_points(im_1,'books_1_'+str(scale))
			[img_2,kp_2,des_2] = opencv_interest_points(im_2,'books_2_'+str(scale))
			opencv_correspondence(img_1,kp_1,des_1,img_2,kp_2,des_2,'books_'+str(scale),10)

	im_1 = './HW4-Images/Figures/fountain_1.jpg'
	im_2 = './HW4-Images/Figures/fountain_2.jpg'
	for scale in scales:
		corner_detection_image_pairs(im_1,im_2,'fountain_1','fountain_2',99,scale)
		if scale==1.2:
			[img_1,kp_1,des_1] = opencv_interest_points(im_1,'fountain_1'+str(scale))
			[img_2,kp_2,des_2] = opencv_interest_points(im_2,'fountain_2'+str(scale))
			opencv_correspondence(img_1,kp_1,des_1,img_2,kp_2,des_2,'fountain_'+str(scale),10)

	im_1 = './HW4-Images/Figures/custom_pair_1_1.jpg'
	im_2 = './HW4-Images/Figures/custom_pair_1_2.jpg'
	for scale in scales:
		corner_detection_image_pairs(im_1,im_2,'custom_pair_1_1','custom_pair_1_2',99.95,scale)

	im_1 = './HW4-Images/Figures/custom_pair_2_1.jpg'
	im_2 = './HW4-Images/Figures/custom_pair_2_2.jpg'
	for scale in scales:
		corner_detection_image_pairs(im_1,im_2,'custom_pair_2_1','custom_pair_2_2',99,scale)
	


main()
