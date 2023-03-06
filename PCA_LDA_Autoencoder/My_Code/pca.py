import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.io
import cv2,os
import heapq
from collections import Counter

def read_data(data_dir,num_classes,image_per_class):
	labels = list()
	for ix in range(1,num_classes+1):
		for jx in range(1,image_per_class+1):
			class_id = str(ix) if ix >= 10 else '0'+str(ix)
			image_id = str(jx) if jx >= 10 else '0'+str(jx)
			image = cv2.imread(data_dir+'/'+class_id+'_'+image_id+'.png')
			image = cv2.resize(image,(32,32),interpolation = cv2.INTER_AREA)
			if len(image.shape)>2:
				image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			if ix==1 and jx==1:
				image_shape = image.shape
			image = np.reshape(image.flatten(),[-1,1])
			if ix==1 and jx==1:
				image_set = image
			else:
				image_set = np.concatenate((image_set,image),axis=1)
			labels.append(ix)
	labels = np.array(labels).squeeze()
	return image_set,labels,image_shape
def center_data(data,save_dir):
	mean_data = np.mean(data,axis=1)
	with open(save_dir+'/image_mean.npy', 'wb') as f:
		np.save(f,mean_data)
	data = data - np.reshape(mean_data,[-1,1])
	return data
def normalize_data(data):
	data_norm = np.linalg.norm(data,axis=0)
	data_norm = np.reshape(data_norm,[1,-1])
	data_norm = np.tile(data_norm,(data.shape[0],1))
	data = np.divide(data,data_norm)
	return data
def covariance_matrix(data_set,save_dir,mode='calc'):
	if mode.upper() == "CALC":
		C = np.matmul(data_set,data_set.T)
		C_compressed = np.matmul(data_set.T,data_set)
		with open(save_dir+'/train_data_covariance_matrix.npy', 'wb') as f:
			np.save(f,C)
			np.save(f,C_compressed)
	elif mode.upper() == "LOAD":
		with open(save_dir+'/train_data_covariance_matrix.npy', 'rb') as f:
			C = np.load(f)
			C_compressed = np.load(f)
	return C, C_compressed
def plot_covariance_matrix(C_matrix,save_dir,save_path):
	fig = plt.figure()
	plt.matshow(C_matrix)
	plt.colorbar()
	plt.title('Covariance Matrix')
	plt.savefig(save_dir+'/'+save_path)
def plot_eigenvectors(eigs,K,save_dir):
	plt.figure()
	plt.plot(eigs[:,:K])
	plt.legend(['E'+str(i) for i in range(1,K+1)])
	plt.savefig(save_dir+'/eigenvectors_K_'+str(K)+'.png')
def plot_eigen_faces(eigs,save_dir,image_shape,num_faces):
	count = 0
	num_rows = int(np.sqrt(num_faces))
	num_cols = int(num_faces/float(num_rows))
	eig_f = np.zeros((image_shape[0]*num_rows,image_shape[1]*num_cols))
	for i in range(0,num_faces):
		row = int(np.floor((i)/num_cols))*image_shape[0]
		col = count*image_shape[1]
		eig_f[row:row+image_shape[0],col:col+image_shape[1]] = np.reshape(eigs[:,i],list(image_shape))
		count += 1
		if count >= num_cols:
			count = 0
	plt.figure()
	plt.imshow(eig_f)
	plt.savefig(save_dir+'/Eigen_Faces.png')
def calculate_K_large_eigenvectors(data,C_matrix_compressed,K,image_shape,save_dir,mode='calc'):
	if mode.upper() == "CALC":
		[w,u] = np.linalg.eig(C_matrix_compressed)#w=eigenvalues, v=eigenvectors; the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
		v = np.matmul(data,u)
		v = np.real(v)
		#normalize eigenvectors
		v = v/np.linalg.norm(v,axis=0)
		with open(save_dir+'/eigs.npy', 'wb') as f:
			np.save(f, w)
			np.save(f, v)
		w = np.abs(w)
		ws = heapq.nlargest(K, w)
		ind = [list(w).index(i) for i in ws]
		Q = v[:,ind]
		with open(save_dir+'/eigenvectors.npy', 'wb') as f:
			np.save(f,Q)
	elif mode.upper() == "LOAD":
		with open(save_dir+'/eigs.npy', 'rb') as f:
			w = np.load(f)
			v = np.load(f)
		with open(save_dir+'/eigenvectors.npy', 'rb') as f:
			Q = np.load(f)
	#plot_eigenvectors(Q,K,save_dir)
	if K==49:
		plot_eigen_faces(np.abs(Q),save_dir,image_shape,K)
	return Q[:,:K],v[:,:K],w[:K]
def get_low_dimensional_projections(v,Q,data,save_dir):
	ak = list()
	bk = list()
	ck = list()
	for i in range(0,data.shape[1]):
		ak.append(np.dot(data[:,i],Q[:,0]))
		bk.append(np.dot(data[:,i],Q[:,1]))
		ck.append(np.dot(data[:,i],Q[:,2]))

	ak,bk,ck = np.array(ak),np.array(bk),np.array(ck)
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(ak,bk,ck)
	ax.set_xlabel('eig_0')
	ax.set_ylabel('eig_1')
	ax.set_zlabel('eig_2')
	plt.savefig(save_dir+'/projectioins_along_first_three_eig.png')
	return ak,bk,ck
def get_gt_coeffs(data,Q,K,image_shape,save_dir):
	coeff = np.matmul(data.T,Q)
	with open(save_dir+'/coefficients.npy', 'wb') as f:
		np.save(f,coeff)
	return coeff
def get_prediction(data,Q,K,image_shape,save_dir):
	with open(save_dir+'/image_mean.npy', 'rb') as f:
		image_mean = np.reshape(np.load(f),[-1,1])

	with open(save_dir+'/coefficients.npy', 'rb') as f:
		coeff_gt = np.load(f)
	data = data - image_mean
	data = normalize_data(data)
	coeff_pred = np.matmul(data.T,Q)
	distance = np.zeros((data.shape[1],coeff_gt.shape[0]))
	for ix in range(0,data.shape[1]):
		distance[ix,:] = np.sqrt(np.sum(np.square(coeff_gt-coeff_pred[ix,:]),axis=1)).T
	return distance
def calculate_accuracy(y_true,distance,image_per_class,num_classes,method="L2",params=None):
	if method.upper() == "L2":
		match_ind = np.argmin(distance,axis=1)+1#+1 is for indices start from 1
		y_pred = y_true[match_ind]
		y_pred = y_pred.astype(int).squeeze()

	elif method.upper() == "KNN":
		num_neighbours = params[0]
		y_pred = list()
		for im in range(0,distance.shape[0]):
			dist = distance[im,:]
			class_hits = np.zeros((num_classes))
			class_dist = np.zeros((num_classes))
			for ix in range(0,num_neighbours):
				min_idx = np.argmin(dist)
				min_label = y_true[min_idx]-1
				class_hits[min_label] += 1
				class_dist[min_label] += dist[min_idx]
				dist[min_idx] = float('inf')
			max_hits = np.max(class_hits)
			class_avg_dist = class_dist/max_hits
			class_avg_dist = np.where(class_avg_dist==0,np.inf,class_avg_dist)
			class_id = np.argmin(class_avg_dist)+1
			y_pred.append(class_id)
		y_pred = np.array(y_pred).astype(int).squeeze()

	match = np.where(y_true==y_pred,1,0)
	correct_labels = np.sum(match)
	accuracy = correct_labels/y_true.shape[0]*100
	return accuracy
def main():
	train_dir = '../FaceRecognition/train'
	test_dir = '../FaceRecognition/test'
	save_dir = './Results/PCA'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	num_classes = 30
	train_image_per_class = 21
	test_image_per_class = 21
	classifier = "KNN"#options = KNN,L2
	params = [1]
	Kmax = 25

	X_train,Y_train,image_shape = read_data(train_dir,num_classes,train_image_per_class)
	X_test,Y_test,_ = read_data(test_dir,num_classes,test_image_per_class)
	X_train = center_data(X_train,save_dir)
	X_train = normalize_data(X_train)
	print('Train data = ',X_train.shape)
	print('Test data = ',X_test.shape)
	C_matrix,C_matrix_compressed = covariance_matrix(X_train,save_dir,mode='calc')
	print('C_matrix_train = ',C_matrix.shape)
	print('C_matrix_compressed_train = ',C_matrix_compressed.shape)
	plot_covariance_matrix(C_matrix,save_dir,'train_data_covariance_matrix.png')
	Accuracies = list()
	for K in range(1,Kmax+1,1):
		[Q,v,w] = calculate_K_large_eigenvectors(X_train,C_matrix_compressed,K,image_shape,save_dir,mode='calc')
		#[ak,bk,ck] = get_low_dimensional_projections(v,Q,X_train,save_dir)
		coeff = get_gt_coeffs(X_train,Q,K,image_shape,save_dir)
		distance = get_prediction(X_test,Q,K,image_shape,save_dir)
		accuracy = calculate_accuracy(Y_test,distance,test_image_per_class,num_classes,method=classifier,params=params)
		print("K = ",K,", Accuracy = ",accuracy,'%')
		Accuracies.append(accuracy)
		with open(save_dir+'/PCA_Accuracies.npy', 'wb') as f:
			np.save(f,Accuracies)
main()
