import cv2
import os
import numpy as np

def image_read(file_path):
	image = cv2.imread(file_path)
	dimensions = image.shape#[height. weidth,channels]
	return image, dimensions

def opencv_interest_points(image,detector,out_file):
	gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	if detector.upper()=="SIFT":
		det = cv2.SIFT_create()
	kp,des = det.detectAndCompute(gray,None)
	img=cv2.drawKeypoints(gray,kp,image)
	cv2.imwrite('sift_keypoints_'+out_file+'.jpg',img)
	return img,kp,des

def opencv_correspondence(img_1,kp_1,des_1,img_2,kp_2,des_2,out_file,N):
	#N 	: how many correspondences to be kept
	bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)# create BFMatcher object
	matches = bf.match(des_1,des_2)# Match descriptors.
	matches = sorted(matches, key = lambda x:x.distance)# Sort them in the order of their distance.
	img3 = cv2.drawMatches(img_1,kp_1,img_2,kp_2,matches[:N], None,flags=2)# Draw first 10 matches.
	cv2.imwrite('opencv_correspondences_'+out_file+'.jpg',img3)
	return img3,matches

def draw_correspondences(img_1,img_2,point_set_1,point_set_2,point_set_3,point_set_4,name):
	stacked_image = np.concatenate((img_1,img_2),1)
	for ix in range(0,point_set_1.shape[0]):
		pt1 = tuple(np.round(point_set_1[ix]).astype(int))
		pt2 = tuple(np.round(np.array(point_set_2[ix] + [img_1.shape[1],0])).astype(int))
		cv2.line(stacked_image,pt1,pt2,(0, 255, 0), 2)
		cv2.circle(stacked_image,pt1,3,(0, 0, 255),-1)
		cv2.circle(stacked_image,pt2,3,(0, 0, 255),-1)
	for ix in range(0,point_set_3.shape[0]):
		pt1 = tuple(np.round(point_set_3[ix]).astype(int))
		pt2 = tuple(np.round(np.array(point_set_4[ix] + [img_1.shape[1],0])).astype(int))
		cv2.line(stacked_image,pt1,pt2,(255, 0, 0), 2)
		cv2.circle(stacked_image,pt1,3,(0, 0, 255),-1)
		cv2.circle(stacked_image,pt2,3,(0, 0, 255),-1)
	cv2.imwrite('Image_Correspondences_'+name+'.jpg',stacked_image)

def cost_function(H,point_set_1,point_set_2):
	Range_Points = list()
	F= list()
	for ix in range(point_set_1.shape[0]):
		Range_Points.append(point_set_2[ix,0])
		Range_Points.append(point_set_2[ix,1])

		f1 = np.array(H[0]*point_set_1[ix,0]+H[1]*point_set_1[ix,1]+H[2] / H[6]*point_set_1[ix,0]+H[7]*point_set_1[ix,1]+H[8])
		f2 = np.array(H[3]*point_set_1[ix,0]+H[4]*point_set_1[ix,1]+H[5] / H[6]*point_set_1[ix,0]+H[7]*point_set_1[ix,1]+H[8])

		F.append(f1)
		F.append(f2)

	return np.array(Range_Points) - np.array(F)

def form_line(pt1,pt2):
	if len(pt1)<3:
		pt1.append(1)
	if len(pt2)<3:
		pt2.append(1)
	t = np.cross(pt1,pt2)
	if t[2]!= 0:
		line = t/t[2]
	else:
		line = t
	return line

def form_intersection(line_1,line_2):
	t = np.cross(line_1,line_2)
	if t[2]!= 0:
		intersection = t/t[2]
	else:
		intersection = t
	return intersection

def stich_frame(panorama,Frame,H,side,previous_min_X=0):
	[height,width,channels] = Frame.shape
	Frame_corners = np.array([[0,0,1],[0,height-1,1],[width-1,height-1,1],[width-1,0,1]])
	temp = np.dot(H,Frame_corners.T)
	Frame_corners_transformed = (np.round(temp/temp[2,:]).astype(int)).T
	frame_top_edge = form_line([0,1],[10,1])
	frame_bottom_edge =form_line([0,panorama.shape[0]],[10,panorama.shape[0]])
	if side.upper() == 'RIGHTSIDE':
		panorama_edge = form_line(Frame_corners_transformed[2,:].tolist(),Frame_corners_transformed[3,:].tolist())
		top_intersection = form_intersection(frame_top_edge,panorama_edge)

		bottom_intersection = form_intersection(frame_bottom_edge,panorama_edge)
		max_X = np.round(max(top_intersection[0],bottom_intersection[0])).astype(int)
		min_X = np.min(Frame_corners_transformed[:,0])
		expanded_panorama = np.zeros((panorama.shape[0],np.abs(max_X-panorama.shape[1]),3),np.uint8)
		print(min_X)
		print(max_X)
		print(panorama.shape[0])
		print(np.abs(max_X-min_X))
		new_frame = np.zeros((panorama.shape[0],np.abs(max_X-min_X),3),np.uint8)
		panorama = np.concatenate((panorama,expanded_panorama),1)
		indices = np.indices((new_frame.shape[1],new_frame.shape[0]))
		print('indices = ',indices.shape)
		x_indices = indices[0,:,:].reshape(new_frame.shape[1]*new_frame.shape[0],1)+min_X
	elif side.upper() == 'LEFTSIDE':
		panorama_edge = form_line(Frame_corners_transformed[0,:].tolist(),Frame_corners_transformed[1,:].tolist())
		top_intersection = form_intersection(frame_top_edge,panorama_edge)
		bottom_intersection = form_intersection(frame_bottom_edge,panorama_edge)
		min_X = np.round(min(top_intersection[0],bottom_intersection[0])).astype(int)
		max_X = np.max(Frame_corners_transformed[:,0])
		expanded_panorama = np.zeros((panorama.shape[0],abs(min_X-previous_min_X),3))
		new_frame = np.zeros((panorama.shape[0],abs(abs(min_X)+max_X),3))
		panorama = np.concatenate((expanded_panorama,panorama),1)
		indices = np.indices((new_frame.shape[1],new_frame.shape[0]))
		x_indices = indices[0,:,:].reshape(new_frame.shape[1]*new_frame.shape[0],1)+min_X
	else:
		print("Please specify the direction of stitching, rightside or leftside")
		return

	y_indices = indices[1,:,:].reshape(new_frame.shape[1]*new_frame.shape[0],1)
	z_indices = np.ones((new_frame.shape[1]*new_frame.shape[0],1),np.uint8)
	final_indices = np.concatenate((x_indices,y_indices,z_indices),1)
	new_indices = (np.dot(np.linalg.pinv(H).astype(float),final_indices.T)).T
	new_indices[:,0] = new_indices[:,0]/new_indices[:,2]
	new_indices[:,1] = new_indices[:,1]/new_indices[:,2]
	new_indices[:,2] = new_indices[:,2]/new_indices[:,2]
	new_indices = new_indices.astype(int)

	final_indices = final_indices[new_indices[:,0] >= 0]
	new_indices = new_indices[new_indices[:,0] >= 0]
	final_indices = final_indices[new_indices[:,1] >= 0]
	new_indices =new_indices[new_indices[:,1] >= 0]
	final_indices = final_indices[new_indices[:,0] < width]
	new_indices = new_indices[new_indices[:,0] < width]
	final_indices = final_indices[new_indices[:,1] < height]
	new_indices =new_indices[new_indices[:,1] < height]

	print('PN = ',panorama.shape)
	print('F = ',Frame.shape)
	print(new_indices.shape)
	if not side.upper()=='Rightside':
		final_indices[:,0] = final_indices[:,0]-min_X
	for ix in range(new_indices.shape[0]):
		panorama[final_indices[ix,1]][final_indices[ix,0]] = Frame[new_indices[ix,1]][new_indices[ix,0]]

	return panorama,min_X
