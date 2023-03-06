import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_disparity_map(image_1,image_2,M,dmax,names,padding_mode='constant'):
	disparity_map = np.zeros(image_1.shape)
	image_1 = np.pad(image_1,((int(M/2),int(M/2)),(int(M/2),int(M/2)), (0,0)),mode=padding_mode)
	image_2 = np.pad(image_2,((int(M/2),int(M/2)),(int(M/2),int(M/2)), (0,0)),mode=padding_mode)
	height,width,channels = image_1.shape
	for ch in range(0,channels):
		for i in range(int(M/2),height-int(M/2)):
			for j in range(int(M/2),width-int(M/2)):
				if j<dmax:
					num_shifts = j-int(M/2)
				else:
					num_shifts = dmax
				num_shifts = num_shifts+1 if num_shifts==0 else num_shifts
				disparity = list()
				window_1 = image_1[i-int(M/2):i+int(M/2)+1,j-int(M/2):j+int(M/2)+1,ch].flatten()
				bitvector_1 = np.array(np.where(window_1>image_1[i,j,ch], 1, 0))
				for k in range(0,num_shifts):
					try:
						row_ind_2 = i
						col_ind_2 = j-k
						window_2 = image_2[row_ind_2-int(M/2):row_ind_2+int(M/2)+1,col_ind_2-int(M/2):col_ind_2+int(M/2)+1,ch]
						window_2 = window_2.flatten()
						bitvector_2 = np.array(np.where(window_2>image_2[row_ind_2,col_ind_2,ch], 1, 0))
						disp = np.sum(np.logical_xor(bitvector_1,bitvector_2).astype(int))
						disparity.append(disp)
					except:
						continue
				disparity = np.array(disparity)
				disparity_map[i-int(M/2),j-int(M/2),ch] = np.min(disparity)
	return disparity_map
def disparity_error(disparity_gt,disparity_est,delta=2):
	ind = disparity_gt.nonzero()
	N = len(ind[0])
	difference = np.abs(disparity_gt-disparity_est)
	accuracy = np.sum(difference[ind] <= delta)/1.0/N
	error_mask = np.zeros(difference.shape)
	error_mask[ind] = difference[ind]<=delta
	return accuracy,error_mask,difference

def main():
	if not os.path.exists('./Results'):
		os.makedirs('./Results')
	#read image pairs
	fname_1 = './Images/im2.png'
	image_1 = cv2.imread(fname_1)
	fname_2 = './Images/im6.png'
	image_2 = cv2.imread(fname_2)

	s1 = fname_1.split('/')[-1]
	s2 = fname_2.split('/')[-1]
	names = [s1.split('.')[0],s2.split('.')[0]]

	fname_1_dsp = './Images/disp2.png'
	fname_2_dsp = './Images/disp6.png'
	disparity_gt = cv2.imread(fname_1_dsp)
	factor = 128.0
	disparity_gt = (disparity_gt.astype(np.float32)/factor).astype(np.int8)
	dmax = np.max(disparity_gt)
	'''
	delta = 2
	Accuracies = list()
	for M in range(3,41,2):
		save_path = './Results/Disparity_Map_M_'+str(M)+'_dmax_'+str(dmax)+'.png'
		disparity_est = calculate_disparity_map(image_1,image_2,M,dmax,names)
		cv2.imwrite(save_path,disparity_est.astype(np.uint8)*16)
		accuracy,error_mask,difference_map = disparity_error(disparity_gt,disparity_est,delta=delta)
		cv2.imwrite('./Results/Disparity_Map_Error_M_'+str(M)+'_dmax_'+str(dmax)+'.png',difference_map)
		cv2.imwrite('./Results/Disparity_Map_Error_Mask_M_'+str(M)+'_dmax_'+str(dmax)+'.png',error_mask.astype(np.uint8)*255)
		print("Disparity Map Calculation for M = ",M,' delta = ',delta,' Accuracy = ',accuracy*100)
		Accuracies.append(accuracy*100)
	np.save('disparity_map_accuracies_delta_'+str(delta)+'_factor_'+str(factor)+'.npy',Accuracies)
	'''
	fig = plt.figure()
	for delta in range(1,8):
		X = np.load('disparity_map_accuracies_delta_'+str(delta)+'.npy')
		plt.plot(X,label='delta = '+str(delta))
	plt.legend()
	plt.xlabel('M')
	plt.ylabel('Accuracy(%)')
	plt.xticks(np.arange(1,20), [str(ix) for ix in range(3,40,2)])
	plt.savefig('./Results/Disparity_Accuracy.png')
	plt.show()
main()
