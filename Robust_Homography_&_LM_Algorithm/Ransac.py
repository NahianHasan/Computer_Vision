import cv2
import numpy as np
import random
import math
import Linear_Least_Square_Homography as LSM

def calculate_inliers_outliers(H,delta,point_set_1,point_set_2):
	transformed_points_1 = np.zeros(point_set_1.shape)
	for ix in range(0, point_set_1.shape[0]):
		tx = np.dot(H,np.array(np.append(point_set_1[ix,:],1)).T)
		transformed_points_1[ix] = np.round(np.array([tx[0]/tx[2],tx[1]/tx[2]]))

	distances = np.sqrt(np.sum((point_set_2 - transformed_points_1)**2,1))
	idx = distances<=delta
	inliers_1 = point_set_1[idx]
	inliers_2 = point_set_2[idx]
	idx = distances>delta
	outliers_1 = point_set_1[idx]
	outliers_2 = point_set_2[idx]
	return inliers_1,inliers_2,outliers_1,outliers_2

def Ransac(Int_points_1,Int_points_2,n,sigma,p,epsilon):
	#n = number of randomly selected correspondences for ransac
	#delta = distance threshold between transformed points and original points
	delta = 100*sigma
	num_correspondences = Int_points_1.shape[0]
	N = math.ceil(math.log(1-p)/math.log(1-(1-epsilon)**n))
	M = math.ceil((1-epsilon)*n)
	Inliers = list()
	Outliers = list()
	Optimum_inlier_len = -1
	Optimum_inliers = []
	Optimum_outliers = []
	#print("M = ",M)
	#print("N = ",N)
	for ix in range(N):
		indx = random.sample(range(num_correspondences), n)
		point_set_1 = Int_points_1[indx,:]
		point_set_2 = Int_points_2[indx,:]
		H = LSM.LLSM(point_set_1,point_set_2)
		inliers_1,inliers_2,outliers_1,outliers_2 = calculate_inliers_outliers(H,delta,point_set_1,point_set_2)
		if len(inliers_1) > Optimum_inlier_len:
			Optimum_inlier_len = len(inliers_1)
			Optimum_inliers_1 = inliers_1
			Optimum_inliers_2 = inliers_2
			Optimum_outliers_1 = outliers_1
			Optimum_outliers_2 = outliers_2

	#print("Optimum Inliers 1 = ",Optimum_inliers_1,'\nOptimum Inliers 2 = ',Optimum_inliers_2)
	return Optimum_inliers_1,Optimum_inliers_2,Optimum_outliers_1,Optimum_outliers_2
