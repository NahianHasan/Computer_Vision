import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy import optimize

def opencv_interest_points(img,name):
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT_create()
	kp,des = sift.detectAndCompute(gray,None)
	img=cv2.drawKeypoints(gray,kp,img)
	cv2.imwrite(name,img)
	return img,kp,des
def opencv_correspondence(img_1,kp_1,des_1,img_2,kp_2,des_2,name,N):
	#N 	: how many correspondences to be kept
	bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)# create BFMatcher object
	matches = bf.match(des_1,des_2)# Match descriptors.
	matches = sorted(matches, key = lambda x:x.distance)# Sort them in the order of their distance.
	img3 = cv2.drawMatches(img_1,kp_1,img_2,kp_2,matches[:N], None,flags=2)# Draw first 10 matches.
	cv2.imwrite(name+'opencv_correspondences.jpg',img3)
def Sum_of_Squared_Differences(Im_1,Im_2,corner_indexes_1,corner_indexes_2,N,padding_mode='constant'):
	#N	: N*N neighbourhood
	I_1 = np.pad(Im_1,((int(N/2),int(N/2)),(int(N/2),int(N/2)),(0,0)),mode=padding_mode)
	I_2 = np.pad(Im_2,((int(N/2),int(N/2)),(int(N/2),int(N/2)),(0,0)),mode=padding_mode)
	Best_Matches = np.zeros((corner_indexes_1.shape[0],1))
	SSD_list = np.zeros((corner_indexes_1.shape[0],1))
	for s1 in range(0,corner_indexes_1.shape[0]):
		SSD = list()
		i = corner_indexes_1[s1,0]
		j = corner_indexes_1[s1,1]
		ix = i+int(N/2)
		jx = j+int(N/2)
		f_1 = I_1[ix-int(N/2):ix+int(N/2),jx-int(N/2):jx+int(N/2),0]
		for s2 in range(0,corner_indexes_2.shape[0]):
			i = corner_indexes_2[s2,0]
			j = corner_indexes_2[s2,1]
			ix = i+int(N/2)
			jx = j+int(N/2)
			f_2 = I_2[ix-int(N/2):ix+int(N/2),jx-int(N/2):jx+int(N/2),0]
			#if np.sum(f_2) !=0:
			SSD.append(np.sum(np.square(f_1.flatten() - f_2.flatten())))
		SSD = np.array(SSD).squeeze()
		ind = np.argsort(SSD)[0]
		Best_Matches[s1,0] = ind
		SSD_list[s1,0] = SSD[ind]
	return Best_Matches,SSD_list
def Normalized_Cross_Correlations(Im_1,Im_2,corner_indexes_1,corner_indexes_2,N,padding_mode='constant'):
	#N	: N*N neighbourhood
	I_1 = np.pad(Im_1,((int(N/2),int(N/2)),(int(N/2),int(N/2)),(0,0)),mode=padding_mode)
	I_2 = np.pad(Im_2,((int(N/2),int(N/2)),(int(N/2),int(N/2)),(0,0)),mode=padding_mode)
	Best_Matches = np.zeros((corner_indexes_1.shape[0],1))
	NCC_list = np.zeros((corner_indexes_1.shape[0],1))
	for s1 in range(0,corner_indexes_1.shape[0]):
		i = corner_indexes_1[s1,0]
		j = corner_indexes_1[s1,1]
		ix = i+int(N/2)
		jx = j+int(N/2)
		f_1 = I_1[ix-int(N/2):ix+int(N/2),jx-int(N/2):jx+int(N/2),0]
		m_1 = np.mean(I_1[ix-int(N/2):ix+int(N/2),jx-int(N/2):jx+int(N/2),0])
		NCC = np.zeros((corner_indexes_2.shape[0],1))
		for s2 in range(0,corner_indexes_2.shape[0]):
			i = corner_indexes_2[s2,0]
			j = corner_indexes_2[s2,1]
			ix = i+int(N/2)
			jx = j+int(N/2)
			f_2 = I_2[ix-int(N/2):ix+int(N/2),jx-int(N/2):jx+int(N/2),0]
			m_2 = np.mean(I_2[ix-int(N/2):ix+int(N/2),jx-int(N/2):jx+int(N/2),0])
			num = np.sum(np.multiply((f_1-m_1),(f_2-m_2)))
			den = np.sqrt(np.multiply(np.sum(np.square(f_1-m_1)),np.sum(np.square(f_2-m_2))))
			if den != 0:
				NCC[s2,0] = num/den
		NCC = NCC.squeeze()
		ind = np.argsort(NCC)[0]
		Best_Matches[s1,0] = ind
		NCC_list[s1,0] = NCC[ind]
	return Best_Matches,NCC_list
def plot_correspondences(I_1,I_2,correspondences,M,name):
	#M	: how many correspondences to show
	CI = np.concatenate((I_1,I_2),axis=1)
	im_width = I_1.shape[1]
	if M<len(correspondences):
		ind = random.sample(np.arange(0,len(correspondences)).tolist(),M)
	else:
		ind = np.arange(0,len(correspondences))
	for ix in ind:
		color_matrix = (random.sample(np.arange(0,255).tolist(),1)[0],random.sample(np.arange(0,255).tolist(),1)[0],random.sample(np.arange(0,255).tolist(),1)[0])
		points_1 = tuple(correspondences[ix][0])
		points_2 = tuple([correspondences[ix][1][0]+im_width,correspondences[ix][1][1]])
		cv2.circle(CI,points_1,5,color_matrix,2)
		cv2.circle(CI,points_2,5,color_matrix,2)
		cv2.line(CI,points_1,points_2,color_matrix,2)
	cv2.imwrite('./Results/'+name+'.jpg',CI)
def physical_to_homogeneous(points):
	#points = n*2
	HC = np.concatenate((points,np.ones((points.shape[0],1))),axis=1)
	return HC.T
def homogeneous_to_physical(points):
	#points = 3*n
	physical_points = points[:,:]/points[-1,:]
	return physical_points[:-1,:].T
def data_normalization(points):
	#points = n*2
	mean_coord = np.mean(points,axis=0)
	distances = np.linalg.norm(points-mean_coord,axis=1)
	mean_distance = np.mean(distances)
	normalizing_factor = np.sqrt(2)/mean_distance
	T = np.zeros((3,3))
	T[0,0] = normalizing_factor
	T[0,2] = -normalizing_factor*mean_coord[0]
	T[1,1] = normalizing_factor
	T[1,2] = -normalizing_factor*mean_coord[1]
	T[2,2] = 1
	HC = physical_to_homogeneous(points)
	HC_normalized = np.matmul(T,HC)
	normalized_points = homogeneous_to_physical(HC_normalized)
	return normalized_points, T
def condition_F(F):
	U,D,V = np.linalg.svd(F)
	print(D)
	D[np.argmin(D)] = 0
	print(D)
	F_conditioned = np.matmul(U,np.matmul(np.diag(D),V))
	#F_conditioned = F_conditioned/F_conditioned[2,2]
	return F_conditioned
def Fundamental_Matrix(point_set_1,point_set_2,T1,T2):
	#estimate the fundamental matrix from interest point pairs
	#point_set_1 = shape:[N,2]
	#point_set_2 = shape:[N,2]
	#T1 = normalizing matrix for point_set_1
	#T2 = normalizing matrix for point_set_2
	A = np.zeros((point_set_1.shape[0],9))
	b = np.zeros((point_set_1.shape[0],1))
	for ix in range(0,point_set_1.shape[0]):
		A[ix,:] = [point_set_2[ix,0]*point_set_1[ix,0], point_set_2[ix,0]*point_set_1[ix,1], point_set_2[ix,0], point_set_2[ix,1]*point_set_1[ix,0], point_set_2[ix,1]*point_set_1[ix,1], point_set_2[ix,1], point_set_1[ix,0], point_set_1[ix,1], 1]
	U,D,V = np.linalg.svd(np.matmul(A.T,A))
	F = V[-1,:]
	F = np.reshape(F,[3,3])
	F = F/F[-1,-1]
	F = condition_F(F)
	F = np.matmul(T2.T,np.matmul(F,T1))
	F = F/F[-1,-1]
	return F
def calculate_epipoles(F):
	U,D,V = np.linalg.svd(F)
	left_epipole = V[-1,:].T
	right_epipole = U[:,-1]
	left_epipole = np.reshape(left_epipole/left_epipole[-1],[3,-1])
	right_epipole = np.reshape(right_epipole/right_epipole[-1],[3,-1])
	return left_epipole,right_epipole
def vector_to_matrix(X):
	X = X.squeeze()
	return np.array([[0,-X[2],X[1]],[X[2],0,-X[0]],[-X[1],X[0],0]])
def estimate_projection_matrices(F,left_epipole,right_epipole):
	P_left = np.concatenate((np.eye(3,dtype=float),np.zeros((3,1))),axis=1)
	temp = np.matmul(vector_to_matrix(right_epipole),F)
	P_right = np.concatenate((temp,right_epipole),axis=1)
	return P_left,P_right
def triangulate_points(point_set_1,point_set_2,P_left,P_right):
	triangulated_points = np.zeros((4,point_set_1.shape[0]))
	for ix in range(0,point_set_1.shape[0]):
		A=np.zeros((4,4))
		A[0,:] = point_set_1[ix,0]*P_left[2,:] - P_left[0,:]
		A[1,:] = point_set_1[ix,1]*P_left[2,:] - P_left[1,:]
		A[2,:] = point_set_2[ix,0]*P_right[2,:] - P_right[0,:]
		A[3,:] = point_set_2[ix,1]*P_right[2,:] - P_right[1,:]
		U,D,V = np.linalg.svd(np.matmul(A.T,A))
		X = V[-1,:]
		X = X/X[-1]
		triangulated_points[:,ix] = X.T
	return triangulated_points
def cost_function(P_right,point_set_1,point_set_2,P_left,triangulated_points):
	P_right = np.reshape(P_right,[3,4])
	reprojected_point_set_1 = np.matmul(P_left,triangulated_points)
	reprojected_point_set_2 = np.matmul(P_right,triangulated_points)
	reprojected_point_set_1 = reprojected_point_set_1[:-1,:]/reprojected_point_set_1[-1,:]
	reprojected_point_set_2 = reprojected_point_set_2[:-1,:]/reprojected_point_set_2[-1,:]
	error_1 = point_set_1-reprojected_point_set_1.T
	error_2 = point_set_2-reprojected_point_set_2.T
	error = np.square(error_1.flatten()) + np.square(error_2.flatten())
	return error
def nonlinear_optimization(point_set_1,point_set_2,P_left,P_right):
	triangulated_points = triangulate_points(point_set_1,point_set_2,P_left,P_right)
	parameters = P_right.flatten()
	lm_refined_P_right = optimize.least_squares(cost_function,parameters,args=[point_set_1,point_set_2,P_left,triangulated_points],method='lm',verbose=1,ftol=1e-15)
	lm_refined_P_right = np.reshape(lm_refined_P_right.x,[3,4])
	lm_refined_P_right = lm_refined_P_right / lm_refined_P_right[-1,-1]
	return P_left,lm_refined_P_right
def fundamental_matrix_from_projection(P_left,P_right):
	right_epipole = P_right[:,-1]
	right_epipole_x = vector_to_matrix(right_epipole)
	P_pseudo = np.matmul(P_left.T, np.linalg.inv(np.matmul(P_left,P_left.T)))
	F = np.matmul(right_epipole_x, np.matmul(P_right,P_pseudo))
	F = F/F[-1,-1]
	return F
def estimate_right_homography(image_2, image_2_pt_normalized, right_epipole_refined, P_right_refined):
	height, width = image_2.shape[0], image_2.shape[1]
	#Translation matrix to translate the image to the center
	T1 = np.array([[1., 0., -width/2.],[0., 1., -height/2.],[0., 0., 1.]])
	#Rotation matrix to rotate the epipolar line to x-axis
	phi = np.arctan((right_epipole_refined[1,0]-height/2.0)/(right_epipole_refined[0,0]-width/2.0))
	phi = -phi
	R = np.array([[np.cos(phi), -np.sin(phi), 0.],[np.sin(phi), np.cos(phi), 0.],[0.,0.,1.]])
	#compute the homography that translates epipole to infinity
	#f = np.linalg.norm(np.array([right_epipole_refined[1,0]-height/2.0, right_epipole_refined[0,0]-width/2.0]))
	f = np.cos(phi)*(right_epipole_refined[0,0]-width/2.0) - np.sin(phi)*(right_epipole_refined[1,0]-height/2.0)
	G = np.eye(3,dtype=float)
	G[2,0] = -1/f
	H_right = np.matmul(G,np.matmul(R,T1))
	#Translate the image back to original location
	img_2_center = transform_image(H_right,np.array([width/2.0,height/2.0]).reshape([1,2]))
	T2 = np.array([[1., 0., width/2.0-img_2_center[0,0]],[0., 1., height/2.0-img_2_center[0,1]],[0., 0., 1.]])
	H_right = np.matmul(T2,H_right)
	H_right = H_right/H_right[-1,-1]
	return H_right
def transform_image(H,input_data):
	input_data = input_data.T
	input_data = np.concatenate((input_data,np.ones((1,input_data.shape[1]))),axis=0)
	transformed_image = np.matmul(H,input_data)
	transformed_image = transformed_image[0:2,:]/transformed_image[2,:]
	return transformed_image[0:2,:].T
def estimate_left_homography(image_1,image_1_pt_normalized, image_2_pt_normalized, left_epipole_refined, P_left_refined, P_right_refined, H_right):
	height, width = image_1.shape[0], image_1.shape[1]
	T1 = np.array([[1., 0., -width/2.],[0., 1., -height/2.],[0., 0., 1.]])
	#Rotation matrix to rotate the epipolar line to x-axis
	phi = np.arctan((left_epipole_refined[1,0]-height/2.0)/(left_epipole_refined[0,0]-width/2.0))
	phi = -phi
	R = np.array([[np.cos(phi), -np.sin(phi), 0.],[np.sin(phi), np.cos(phi), 0.],[0.,0.,1.]])
	#compute the homography that translates epipole to infinity
	f = np.cos(phi)*(left_epipole_refined[0,0]-width/2.0) - np.sin(phi)*(left_epipole_refined[1,0]-height/2.0)
	G = np.eye(3,dtype=float)
	G[2,0] = -1/f
	H0 = np.matmul(G,np.matmul(R,T1))
	H0 = H0/H0[-1,-1]
	image_2_pt_projected = transform_image(H_right,image_2_pt_normalized)
	image_1_pt_projected = transform_image(H0,image_1_pt_normalized)
	A = np.zeros((image_1_pt_projected.shape[0],3))
	b = np.zeros((image_1_pt_projected.shape[0],1))
	for ix in range(0,image_1_pt_projected.shape[0]):
		A[ix,:] = [image_1_pt_projected[ix,0],image_1_pt_projected[ix,1],1.]
		b[ix,0] = image_2_pt_projected[ix,0]
	h = np.matmul(np.linalg.pinv(A),b)
	h = h.squeeze()
	HA = np.array([[h[0],h[1],h[2]],[0.,1.,0.],[0.,0.,1.]])
	H_left = np.matmul(HA,H0)
	img_1_center = transform_image(H_left,np.array([width/2.0,height/2.0]).reshape([1,2]))
	T2 = np.array([[1., 0., width/2.0-img_1_center[0,0]],[0., 1., height/2.0-img_1_center[0,1]],[0., 0., 1.]])
	H_left = np.matmul(T2,H_left)
	H_left = H_left/H_left[-1,-1]
	return H_left
def plot_rectified_images(image_1_rectified,image_2_rectified,save_paths):
	cv2.imwrite(save_paths[0], image_1_rectified)
	cv2.imwrite(save_paths[1], image_2_rectified)
def plot_epipolar_lines(img_1,img_2,point_set_1,point_set_2,F,save_paths):
	image_1 = img_1.copy()
	image_2 = img_2.copy()
	for ix in range(point_set_1.shape[0]):
		cv2.circle(image_1,tuple(point_set_1[ix,:]), 5, color=(0,0,255),thickness=5)
	for ix in range(point_set_2.shape[0]):
		cv2.circle(image_2,tuple(point_set_2[ix,:]), 5, color=(0,0,255),thickness=5)

	epilines_left = np.matmul(F.T,physical_to_homogeneous(point_set_2)).T
	epilines_right = np.matmul(F,physical_to_homogeneous(point_set_1)).T

	for x,y,w in epilines_left:
		cv2.line(image_1,(0,int(-w/y)), (image_1.shape[1]-1,int(-(w+x*(image_1.shape[1]-1))/y)), color=(0,0,0), thickness=3)
	for x,y,w in epilines_right:
		cv2.line(image_2,(0,int(-w/y)), (image_2.shape[1]-1,int(-(w+x*(image_2.shape[1]-1))/y)), color=(0,0,0), thickness=3)

	cv2.imwrite(save_paths[0],image_1)
	cv2.imwrite(save_paths[1],image_2)
def image_rectification(image_1_pt,image_2_pt,image_1,image_2,fnames,names,canvas_size):
	##### Step 1 - Estimating the Fundamental Matrix F
	print("Step 1 - Estimating the Fundamental Matrix F")
	image_1_pt_normalized,T1 = data_normalization(image_1_pt)
	image_2_pt_normalized,T2 = data_normalization(image_2_pt)
	F = Fundamental_Matrix(image_1_pt_normalized,image_2_pt_normalized,T1,T2)
	print("Fundamental Matrix, F = ")
	print(F)
	print("det(F) = ", np.linalg.det(F))
	print("Rank(F) = ",np.linalg.matrix_rank(F))

	##### Step 2 - Estimating epipoles
	print('\n\n')
	print("Step 2 - Estimating epipoles")
	left_epipole, right_epipole = calculate_epipoles(F)
	print("Left Epipole = \n",left_epipole)
	print("Right Epipole = \n",right_epipole)

	##### Step 3 - Estimating initial projection matrices
	print('\n\n')
	print("Step 3 - Estimating initial projection matrices")
	P_left,P_right = estimate_projection_matrices(F,left_epipole,right_epipole)
	print("Left projection matrix = \n",P_left)
	print("Right projection matrix = \n",P_right)

	##### Step 4 - Refine the right projection matrix
	print('\n\n')
	print("Step 4 - Refine the right projection matrix")
	P_left_refined,P_right_refined = nonlinear_optimization(image_1_pt_normalized,image_2_pt_normalized,P_left,P_right)
	print("Refined Left projection matrix = \n",P_left_refined)
	print("Refined projection matrix = \n",P_right_refined)

	##### Step 5 - Refine the fundamental matrix
	print('\n\n')
	print("Step 5 - Refine the fundamental matrix")
	F_refined = fundamental_matrix_from_projection(P_left_refined,P_right_refined)
	print("F (refined) = \n",F_refined)
	print("det(F_refined) = ", np.linalg.det(F_refined))
	print("Rank(F_refined) = ",np.linalg.matrix_rank(F_refined))
	left_epipole_refined, right_epipole_refined = calculate_epipoles(F_refined)
	print("Left Epipole (refined) = \n",left_epipole_refined)
	print("Right Epipole (refined) = \n",right_epipole_refined)

	##### Step 6 - Estimate the Right Homography matrix
	print('\n\n')
	print("Step 6 - Estimate the Right Homography matrix")
	H_right = estimate_right_homography(image_2, image_2_pt_normalized, right_epipole_refined, P_right_refined)
	print("Right image homography for rectification= \n",H_right)

	##### Step 7 - Estimate the Left Homography matrix
	print('\n\n')
	print("Step 7 - Estimate the Left Homography matrix")
	H_left = estimate_left_homography(image_1,image_1_pt_normalized, image_2_pt_normalized, left_epipole_refined, P_left_refined, P_right_refined, H_right)
	print("Left image homography for rectification= \n",H_left)

	##### Step 8 - Apply homography to images for rectification
	print('\n\n')
	print("Step 8 - Apply homography to images for rectification")
	image_1 = cv2.imread(fnames[0])
	image_2 = cv2.imread(fnames[1])
	image_1_rectified = cv2.warpPerspective(image_1,H_left,tuple(canvas_size))
	image_2_rectified = cv2.warpPerspective(image_2,H_right,tuple(canvas_size))
	print("Step 8 complete")

	######## Save the rectified images
	print("Saving rectified images")
	save_paths = ['./Results/'+names[0]+'_rectified.png','./Results/'+names[1]+'_rectified.png']
	plot_rectified_images(image_1_rectified,image_2_rectified,save_paths)
	save_paths = ['./Results/'+names[0]+'_epilines.png','./Results/'+names[1]+'_epilines.png']
	image_1 = cv2.imread(fnames[0])
	image_2 = cv2.imread(fnames[1])
	plot_epipolar_lines(image_1,image_2,image_1_pt,image_2_pt,F_refined,save_paths)


	##### Plot the rectified images along with correspondences
	pnts1h = transform_image(H_left,image_1_pt).astype(int)
	pnts2h = transform_image(H_right,image_2_pt).astype(int)
	correspondences = [[pnts1h[ix].tolist(),pnts2h[ix].tolist()] for ix in range(pnts1h.shape[0])]
	plot_correspondences(image_1_rectified,image_2_rectified,correspondences,40,names[0].split('_')[0]+'_Rectified_Correspondences')
	correspondences = [[image_1_pt[ix].tolist(),image_2_pt[ix].tolist()] for ix in range(image_1_pt.shape[0])]
	plot_correspondences(image_1,image_2,correspondences,40,names[0].split('_')[0]+'_Original_Correspondences')
	#save_paths = ['./Results/'+names[0]+'_rectified_epilines.png','./Results/'+names[1]+'_rectified_epilines.png']
	#plot_epipolar_lines(image_1_rectified,image_2_rectified,pnts1h,pnts2h,F_refined,save_paths)


	return P_left_refined, P_right_refined, F_refined, H_left, H_right, left_epipole_refined, right_epipole_refined, image_1_rectified, image_2_rectified
def interest_point_detection(image_1,image_2,fnames,save_paths,names):

	low_thd=2000
	high_thd=5000
	aperture=5
	num_points = 10000
	best_matching = 1000
	max_search_radius = 5
	edges_1 = cv2.Canny(image_1,low_thd,high_thd,apertureSize=5)
	edges_2 = cv2.Canny(image_2,low_thd,high_thd,apertureSize=5)
	cv2.imwrite(save_paths[0]+'_canny.png', edges_1)
	cv2.imwrite(save_paths[1]+'_canny.png', edges_2)
	points_edges_1 = np.array(np.nonzero(edges_1)).T
	points_edges_2 = np.array(np.nonzero(edges_2)).T
	correspondence_pairs = list()
	SSD_metrics = list()
	point_set_1 = list()
	point_set_2 = list()

	for ix in range(0,points_edges_1.shape[0]):
		#print(ix,'/',points_edges_1.shape[0])
		min_row = points_edges_1[ix,1] - max_search_radius
		max_row = points_edges_1[ix,1] + max_search_radius
		search_cols = np.argwhere((points_edges_2[:,1] >= min_row) & (points_edges_2[:,1] <= max_row)).squeeze()
		search_points_2 = points_edges_2[search_cols,:]
		query_points_1 = np.reshape(points_edges_1[ix,:],[-1,2])

		try:
			if query_points_1[0,0] != 0:
				Best_Matches,SSD_metric = Sum_of_Squared_Differences(image_1,image_2,query_points_1,search_points_2,40)
				Best_Matches = int(Best_Matches.squeeze())
				SSD_metric = SSD_metric.squeeze()
				correspondence_pairs.append([[query_points_1[0,0],query_points_1[0,1]],[search_points_2[Best_Matches,0],search_points_2[Best_Matches,1]]])
				point_set_1.append([query_points_1[0,0],query_points_1[0,1]])
				point_set_2.append([search_points_2[Best_Matches,0],search_points_2[Best_Matches,1]])
				SSD_metrics.append(SSD_metric)
		except Exception as ex:
			#print(ex)
			continue
	best_SSD = np.argsort(SSD_metrics)
	Best_correspondence_list = [correspondence_pairs[best_SSD[ix]] for ix in range(0,best_matching)]
	plot_correspondences(image_1,image_2,Best_correspondence_list,200,names[0].split('_')[0]+'_SSD_correspondence')
	point_set_1 = np.array(point_set_1)
	point_set_2 = np.array(point_set_2)


	[img_1,kp_1,des_1] = opencv_interest_points(image_1,save_paths[0]+'_opencv.png')
	[img_2,kp_2,des_2] = opencv_interest_points(image_2,save_paths[1]+'_opencv.png')
	opencv_correspondence(img_1,kp_1,des_1,img_2,kp_2,des_2,'./Results/',20)

	return point_set_1,point_set_2

def projective_reconstruction(image_1_pt,image_2_pt,image_1,image_2,H_left,H_right,fnames,names):
	image_1_pt = transform_image(np.linalg.inv(H_left),image_1_pt)
	image_2_pt = transform_image(np.linalg.inv(H_right),image_2_pt)
	##### Step 1 - Estimating the Fundamental Matrix F
	image_1_pt_normalized,T1 = data_normalization(image_1_pt)
	image_2_pt_normalized,T2 = data_normalization(image_2_pt)
	F = Fundamental_Matrix(image_1_pt_normalized,image_2_pt_normalized,T1,T2)
	##### Step 2 - Estimating epipoles
	left_epipole, right_epipole = calculate_epipoles(F)
	##### Step 3 - Estimating initial projection matrices
	P_left,P_right = estimate_projection_matrices(F,left_epipole,right_epipole)
	##### Step 4 - Refine the right projection matrix
	P_left_refined,P_right_refined = nonlinear_optimization(image_1_pt_normalized,image_2_pt_normalized,P_left,P_right)
	##### Step 5 - Refine the fundamental matrix
	F_refined = fundamental_matrix_from_projection(P_left_refined,P_right_refined)
	left_epipole_refined, right_epipole_refined = calculate_epipoles(F_refined)
	triangulated_points = triangulate_points(image_1_pt_normalized,image_2_pt_normalized,P_left_refined,P_right_refined)
	return P_left_refined, P_right_refined, F_refined, left_epipole_refined, right_epipole_refined, triangulated_points
def corresponding_points(image_1,image_2,mode='auto'):
	if mode.upper() == "MANUAL":
		#points for given images
		#image_1_pt = np.array([[276,135],[385,191],[197,193],[190,272],[367,43],[447,130],[64,318],[275,341]])
		#image_2_pt = np.array([[242,136],[348,191],[167,192],[158,272],[345,44],[416,130],[31,318],[242,341]])
		#points for custom images
		#image_1_pt = np.array([[73,58],[193,17],[285,160],[392,66],[118,224],[267,356],[354,246],[371,135]])
		#image_2_pt = np.array([[42,86],[160,30],[287,157],[366,54],[107,252],[281,358],[348,231],[360,122]])
		#Stapler Points
		image_1_pt = np.array([[111,290],[82,222],[149,131],[129,146],[112,165],[357,359],[397,336],[401,301],[308,286],[354,267],[368,248],[372,239],[353,212],[385,173],[395,152],[386,120],[170,255]])
		image_2_pt = np.array([[118,242],[96,175],[168,102],[148,111],[126,126],[319,338],[367,324],[384,295],[290,268],[319,250],[344,234],[355,225],[320,189],[357,156],[377,139],[369,103],[165,215]])
	return image_1_pt,image_2_pt
def plot_3D_visualization(fnames,image_1_pt,image_2_pt,triangulated_points,names):
	image_1 = cv2.imread(fnames[0])
	image_2 = cv2.imread(fnames[1])
	for ix in range(0,image_1_pt.shape[0]):
		cv2.circle(image_1,tuple(image_1_pt[ix]),5,(0,0,255),2)
	for ix in range(0,image_2_pt.shape[0]):
		cv2.circle(image_2,tuple(image_2_pt[ix]),5,(0,0,255),2)

	fig = plt.figure()
	ax1 = fig.add_subplot(2, 2, 1)
	plt.imshow(cv2.cvtColor(image_1,cv2.COLOR_BGR2RGB))
	plt.axis('off')
	ax2 = fig.add_subplot(2, 2, 4)
	plt.imshow(cv2.cvtColor(image_2,cv2.COLOR_BGR2RGB))
	plt.axis('off')
	ax3 = fig.add_subplot(2, 2, 3, projection='3d')
	x_data = triangulated_points[0,:]
	y_data = triangulated_points[1,:]
	z_data = triangulated_points[2,:]
	ax3.scatter(x_data,y_data,z_data,s=50)
	'''
	transFigure = fig.transFigure.inverted()
	for ix in range(0,image_1_pt.shape[0]):
		subfig_1_pt = transFigure.transform(ax1.transData.transform(image_1_pt[ix]+[0]))
		subfig_2_pt = transFigure.transform(ax1.transData.transform([x_data[ix],y_data[ix],z_data[ix]]))
		line = matplotlib.lines.Line3D((subfig_1_pt[0],subfig_2_pt[0]),(subfig_1_pt[1],subfig_1_pt[1]),(subfig_1_pt[2],subfig_1_pt[2]), transform=fig.transFigure)
		fig.lines = line
	'''

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig('./Results/'+names[0].split('_')[0]+'_3D_plot.png')
	plt.show()
def main():
	canvas_size = [600,420]
	if not os.path.exists('./Results'):
		os.makedirs('./Results')
	#read image pairs
	fname_1 = './Images/stapler_1.png'
	image_1 = cv2.imread(fname_1)
	fname_2 = './Images/stapler_2.png'
	image_2 = cv2.imread(fname_2)
	if len(image_1.shape)>2:
		image_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
	if len(image_2.shape)>2:
		image_2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)

	s1 = fname_1.split('/')[-1]
	s2 = fname_2.split('/')[-1]
	names = [s1.split('.')[0],s2.split('.')[0]]
	image_1_pt,image_2_pt = corresponding_points(image_1,image_2,mode='manual')
	P_left_refined, P_right_refined, F_refined, H_left, H_right, left_epipole_refined, right_epipole_refined, image_1_rectified, image_2_rectified = image_rectification(image_1_pt,image_2_pt,image_1,image_2,[fname_1,fname_2],names,canvas_size)
	save_paths = ['./Results/'+names[0]+'_int_points','./Results/'+names[1]+'_int_points']
	print('\n\nDetecting interest points')
	image_1_pt_new,image_2_pt_new = interest_point_detection(image_1_rectified,image_2_rectified,[fname_1,fname_2],save_paths,names)
	print('\n\nProjective Distortion')
	P_left_refined_new, P_right_refined_new, F_refined_new, left_epipole_refined_new, right_epipole_refined_new, triangulated_points_new = projective_reconstruction(image_1_pt_new,image_2_pt_new,image_1_rectified,image_2_rectified, H_left, H_right, [fname_1,fname_2],names)
	## 3D visualization
	triangulated_points_original = triangulate_points(image_1_pt,image_2_pt,P_left_refined_new,P_right_refined_new)
	print(triangulated_points_original.shape)
	plot_3D_visualization([fname_1,fname_2],image_1_pt,image_2_pt,triangulated_points_original,names)


main()
