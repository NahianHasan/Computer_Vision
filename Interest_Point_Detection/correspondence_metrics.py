import numpy as np
import cv2

def Sum_of_Squared_Differences(Im_1,Im_2,corner_indexes_1,corner_indexes_2,N,padding_mode='constant'):
	#N	: N*N neighbourhood
	I_1 = np.pad(Im_1,((int(N/2),int(N/2)),(int(N/2),int(N/2)),(0,0)),mode=padding_mode)
	I_2 = np.pad(Im_2,((int(N/2),int(N/2)),(int(N/2),int(N/2)),(0,0)),mode=padding_mode)
	Best_Matches = np.zeros((corner_indexes_1.shape[0],1))
	for s1 in range(0,corner_indexes_1.shape[0]):
		SSD = np.zeros((corner_indexes_2.shape[0],1))
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
			SSD[s2,0] = np.sum(np.square(f_1 - f_2))
		Best_Matches[s1,0] = np.argsort(SSD)[0]
	return Best_Matches

def Normalized_Cross_Correlations(Im_1,Im_2,corner_indexes_1,corner_indexes_2,N,padding_mode='constant'):
	#N	: N*N neighbourhood
	I_1 = np.pad(Im_1,((int(N/2),int(N/2)),(int(N/2),int(N/2)),(0,0)),mode=padding_mode)
	I_2 = np.pad(Im_2,((int(N/2),int(N/2)),(int(N/2),int(N/2)),(0,0)),mode=padding_mode)
	Best_Matches = np.zeros((corner_indexes_1.shape[0],1))

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
		Best_Matches[s1,0] = np.argsort(NCC)[0]

	return Best_Matches

def plot_correspondences(I_1,I_2,Best_Matches,corner_indexes_1,corner_indexes_2,M,name):
	#M	: how many correspondences to show
	CI = np.concatenate((I_1,I_2),axis=1)
	im_width = I_1.shape[1]
	for ix in range(0,min(M,corner_indexes_1.shape[0])):
		points_1 = tuple([corner_indexes_1[ix,1],corner_indexes_1[ix,0]])
		pt2 = corner_indexes_2[int(Best_Matches[ix]),:]
		points_2 = tuple([pt2[1]+im_width,pt2[0]])
		#print(points_1,'---',Best_Matches[ix,],'----',points_2,'----',im_width)
		cv2.circle(CI,points_1,5,(255, 255, 0),1)
		cv2.circle(CI,points_2,5,(255, 255, 0),1)
		cv2.line(CI,points_1,points_2,(255, 255, 0),1)
	cv2.imwrite('Image_Correspondences_'+name+'.jpg',CI)
