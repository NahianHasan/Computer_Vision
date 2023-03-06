import os
import utils as UL
import Linear_Least_Square_Homography as LSM
import numpy as np
import Ransac as RN
from scipy import optimize
import cv2

def Main():
	detector = "sift"
	num_Images = 5
	Image_List = list()
	H_list = np.zeros((num_Images,3,3))#list of pairwise homographies
	image_category = ""

	for kx in range(0,num_Images-1):
		img_file_0 = './HW_5_Images/'+image_category+"_"+str(kx)+'.jpg'
		img_file_1 = './HW_5_Images/'+image_category+"_"+str(kx+1)+'.jpg'

		img_0,_ = UL.image_read(img_file_0)
		img_1,_ = UL.image_read(img_file_1)


		int_points_img_0,kp_0,des_0 = UL.opencv_interest_points(img_0,detector,'img_'+image_category+"_"+str(kx))
		int_points_img_1,kp_1,des_1 = UL.opencv_interest_points(img_1,detector,'img_'+image_category+"_"+str(kx+1))

		correspondence_img,correspondences = UL.opencv_correspondence(int_points_img_0,kp_0,des_0,int_points_img_1,kp_1,des_1,image_category+"_"+str(kx)+'_'+str(kx+1),25)

		corresponding_points = np.zeros((len(correspondences),2))
		for ix in range(0,len(correspondences)):
			corresponding_points[ix,0] = correspondences[ix].queryIdx
			corresponding_points[ix,1] = correspondences[ix].trainIdx

		corresponding_kp_0 = np.zeros((len(correspondences),2))
		corresponding_kp_1 = np.zeros((len(correspondences),2))
		for ix in range(0,corresponding_points.shape[0]):
			corresponding_kp_0[ix,:] = kp_0[int(corresponding_points[ix,0])].pt
			corresponding_kp_1[ix,:] = kp_1[int(corresponding_points[ix,1])].pt

		H = LSM.LLSM(corresponding_kp_0,corresponding_kp_1)
		Optimum_inliers_1,Optimum_inliers_2,Optimum_outliers_1,Optimum_outliers_2 = RN.Ransac(corresponding_kp_0,corresponding_kp_1,20,2,0.99,0.25)
		UL.draw_correspondences(img_0,img_1,Optimum_inliers_1,Optimum_inliers_2,Optimum_outliers_1,Optimum_outliers_2,image_category+"_"+str(kx)+'_'+str(kx+1))

		#Optimize least squares
		H = LSM.LLSM(Optimum_inliers_1,Optimum_inliers_2)
		H = optimize.least_squares(UL.cost_function,np.reshape(H,[1,9]).squeeze(),args=[Optimum_inliers_1,Optimum_inliers_2],method='lm')
		H = np.reshape(H.x,[3,3]).squeeze()

		#print("H matrx "+str(kx)+'_'+str(kx+1),H)
		H_list[kx] = H
	#Add the last image in the list
	for kx in range(0,num_Images):
		img,_ = UL.image_read('./HW_5_Images/'+image_category+"_"+str(kx)+'.jpg')
		Image_List.append(img)

	stitchy=cv2.Stitcher.create()
	(dummy,output)=stitchy.stitch(Image_List)
	cv2.imshow(image_category+"_"+'cv2_panaroma',output)
	cv2.waitKey(0)
	cv2.imwrite(image_category+"_"+'cv2_panorama.jpg',output)
	#Transform H matrices to the anchor image

	anchor_img = 3
	H_to_mid = np.eye(3)
	for ix in range(anchor_img,len(Image_List)):
		H_to_mid = np.matmul(H_to_mid,np.linalg.pinv(H_list[ix]))
		H_list[ix] = H_to_mid
	H_to_mid = np.eye(3)
	for ix in range(anchor_img-1,-1,-1):
		H_to_mid = np.matmul(H_to_mid,H_list[ix])
		H_list[ix] = H_to_mid
	H_list = np.insert(H_list,anchor_img,np.eye(3),0)

	print(H_list)
	#Create panorama
	print(len(Image_List))
	panorama = Image_List[anchor_img-1]
	for ix in range(anchor_img,num_Images):
		panorama,_ = UL.stich_frame(panorama,Image_List[ix],H_list[ix],'Rightside')
		cv2.imwrite(image_category+"_"+"Panorama_right_"+"%d.jpg"%(ix),panorama)
	previous_min_X = 0
	for ix in range(anchor_img-1,-1,-1):
		panorama,previous_min_X = UL.stich_frame(panorama,Image_List[ix],H_list[ix],'Leftside',previous_min_X)
		cv2.imwrite(image_category+"_"+"Panorama_left_"+"%d.jpg"%(ix),panorama)
Main()
