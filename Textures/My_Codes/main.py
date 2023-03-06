import os
import cv2
import lbp
import random
import numpy as np
from joblib import Parallel, delayed
import vgg
from skimage import io, transform
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def prepare_LBP_histograms(ix,images,dir,LBP_SAVE_DIR,name,params):
	R,P,vb = params
	if images[ix][-4:]=='.jpg':
		save_file_name = LBP_SAVE_DIR+'/'+name+'/'+images[ix][:-4]+'.npz'
		if not os.path.exists(save_file_name):
			try:
				print(name,'---R=',R,'---',ix+1,'/',len(images),'---',images[ix])
				im = cv2.imread(dir+'/'+images[ix])
				im = cv2.resize(im,(64,64),interpolation=cv2.INTER_AREA)
				LBP_HISTOGRAM = lbp.lbp(im,R=R,P=P,verbose=vb)
				total_count=0
				for key,val in LBP_HISTOGRAM.items():
					total_count = total_count + val
				for key,val in LBP_HISTOGRAM.items():
					LBP_HISTOGRAM[key] = round(val/total_count,3)
				np.savez(save_file_name,LBP_HISTOGRAM=LBP_HISTOGRAM)
			except Exception as ex:
				print('Problem in reading file - ',images[ix],'---',ex)

def prepare_GM_features(image,model,dir,GM_SAVE_DIR,name,idx,NI):
	save_file_name = GM_SAVE_DIR+'/'+name+'/'+image[:-4]+'.npz'
	if not os.path.exists(save_file_name):
		try:
			print(name,'---',idx+1,'/',NI,'---',image)
			im = cv2.imread(dir+'/'+image)
			im = cv2.resize(im,(256,256),interpolation=cv2.INTER_AREA)
			F = model(im) #Get the feature vector
			F = np.reshape(F,[F.shape[0],F.shape[1]*F.shape[2]])
			G = np.matmul(F,F.T)#calculate the gram matrix
			G = G/np.max(G)#Normalize the gram matrix
			G = np.reshape(G,[1,np.prod(G.shape)])
			np.savez(save_file_name,GM_FEATURE_VECTOR=G)
		except Exception as ex:
			print('Problem in reading file - ',image,'---',ex)

def prepare_AdaIN_features(image,model,dir,AdaIN_SAVE_DIR,name,idx,NI):
	save_file_name = AdaIN_SAVE_DIR+'/'+name+'/'+image[:-4]+'.npz'
	if not os.path.exists(save_file_name):
		try:
			print(name,'---',idx+1,'/',NI,'---',image)
			im = cv2.imread(dir+'/'+image)
			im = cv2.resize(im,(256,256),interpolation=cv2.INTER_AREA)
			F = model(im) #Get the feature vector
			F = np.reshape(F,[F.shape[0],F.shape[1]*F.shape[2]])
			mean_per_channel = np.mean(F,axis=1)
			variance_per_channel = np.var(F,axis=1)
			AD = np.stack((mean_per_channel,variance_per_channel),axis=1)
			AD = AD/np.max(AD)
			AD = np.reshape(AD,[1,np.prod(AD.shape)]).squeeze()
			np.savez(save_file_name,AdaIN_FEATURE_VECTOR=AD)
		except Exception as ex:
			print('Problem in reading file - ',image,'---',ex)

def plot_GM(G,name,eps=1e-14):
	if not os.path.exists('./Results/GM_Matrices'):
		os.makedirs('./Results/GM_Matrices')
	G = G+eps#to avoid nans in the log scale
	G = np.log10(G)
	plt.figure()
	plt.rcParams.update({'font.size': 16})
	plt.imshow(G)
	plt.colorbar()
	plt.xticks(fontweight=600,fontsize=15)
	plt.yticks(fontweight=600,fontsize=15)
	plt.tight_layout()
	plt.savefig('./Results/GM_Matrices/'+name+'.png')

def plot_LBP(LBP_Features,LBP_Labels,classes,R):
	for ix in range(1,len(classes)+1):
		ind = np.argwhere(LBP_Labels==ix).squeeze()
		Class_LBP = LBP_Features[ind,:]
		Mean_Class_LBP = np.mean(Class_LBP,axis=0)
		plt.figure()
		plt.bar(np.arange(0,10).tolist(),Mean_Class_LBP.tolist())
		plt.xticks(fontweight=600,fontsize=15)
		plt.yticks(fontweight=600,fontsize=15)
		plt.xlabel('LBP Histogram Encoding Level (R='+str(R)+')',fontweight=800,fontsize=15)
		plt.ylabel('Mean Probability ('+classes[ix-1]+')',fontweight=800,fontsize=15)
		plt.tight_layout()
		plt.savefig('./Results/Mean_LBP_'+classes[ix-1]+'_R_'+str(R)+'.png')

		plt.figure()
		plt.bar(np.arange(0,10).tolist(),Class_LBP[1,:].tolist())
		plt.xticks(fontweight=600,fontsize=15)
		plt.yticks(fontweight=600,fontsize=15)
		plt.xlabel('LBP Histogram Encoding Level (R='+str(R)+')',fontweight=800,fontsize=15)
		plt.ylabel('Probability ('+classes[ix-1]+')',fontweight=800,fontsize=15)
		plt.tight_layout()
		plt.savefig('./Results/LBP_'+classes[ix-1]+'_R_'+str(R)+'.png')

def data_load(dir,mode,classes,params):
	D = os.listdir(dir)
	FEATURES = list()
	LABELS = list()
	COUNTS = {tx:0 for tx in range(1,len(classes)+1)}
	for ix in range(0,len(D)):
		X = np.load(dir+'/'+D[ix],allow_pickle=True)
		if mode.upper() == 'LBP':
			V = X['LBP_HISTOGRAM'].item()
			temp = np.array([V[k] for k in range(0,10)])
			temp = np.reshape(temp,[1,10])
			FEATURES.append(temp)
		elif mode.upper() == 'GM':
			G = X['GM_FEATURE_VECTOR']
			if params[0] > 0:
				G = G[0,:params[0]]
				G = np.reshape(G,[1,G.shape[0]])
			FEATURES.append(G)
		elif mode.upper() == 'ADAIN':
			A = X['AdaIN_FEATURE_VECTOR']
			A = np.reshape(A,[1,A.shape[0]])
			if params[0] > 0:
				A = A[0,:params[0]]
				A = np.reshape(A,[1,A.shape[0]])
			FEATURES.append(A)
		else:
			print('Please provide a valid moode, either LBP/GM/AdaIN')
		temp = [1 if classes[jx] in D[ix] else 0 for jx in range(0,len(classes))]
		class_id = np.argmax(temp)+1
		COUNTS[class_id] += 1
		LABELS.append(class_id)

	FEATURES = np.array(FEATURES)
	FEATURES = np.reshape(FEATURES,[FEATURES.shape[0],FEATURES.shape[2]])
	LABELS = np.array(LABELS)
	return FEATURES,LABELS,COUNTS

def train(train_dir,test_dir,mode,classes,FEATURE_VECTOR_LENGTH,verbose=False,plot_cfm=False,params=['']):
	X_train,Y_train,COUNTS = data_load(train_dir,mode,classes,[FEATURE_VECTOR_LENGTH])
	#print('AdaIN Training Features = ',X_train_AdaIN.shape)
	#print('AdaIN Training Labels = ',Y_train_AdaIN.shape)
	#print('AdaIN Training Labels Distribution = ',COUNTS)
	P = svm.SVC(decision_function_shape='ovo')
	P.fit(X_train,Y_train)
	X_test,Y_test,COUNTS = data_load(test_dir,mode,classes,[FEATURE_VECTOR_LENGTH])
	#print('AdaIN Testing Features = ',X_test_AdaIN.shape)
	#print('AdaIN Testing Labels = ',Y_test_AdaIN.shape)
	#print('AdaIN Testing Labels Distribution = ',COUNTS)
	YP_test = P.predict(X_test)
	ACCURACY = P.score(X_test,Y_test)
	if verbose:
		print(mode,' Accuracy, featurevector length = ',FEATURE_VECTOR_LENGTH,',    ',mode,' Accuracy = ',np.round(ACCURACY*100,2),'%')
	if plot_cfm:
		#plot_confusion_matrix(P,X_test,Y_test,class_names=classes)
		fig = plt.figure()
		plt.rcParams.update({'font.size': 16})
		cm = confusion_matrix(Y_test, YP_test)
		cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
		P = cmd.plot()
		plt.xticks(fontweight=600,fontsize=16)
		plt.yticks(fontweight=600,fontsize=16)
		plt.xlabel('Predicted Label',fontweight=800,fontsize=16)
		plt.ylabel('True Label',fontweight=800,fontsize=16)
		plt.tight_layout()
		plt.savefig('./Results/Confusion_Matrix_'+mode+params[0]+'.png')
	return [X_train, Y_train, X_test, Y_test, ACCURACY]


def plot_accuracy(fnames,mode,xlabel=None):
	fig = plt.figure()
	M = np.load(fnames[0])
	Acc = M[mode+'_Accuracies'].tolist()
	X = M['FEATURE_VECTOR_LENGTH'].tolist()
	plt.plot(X,Acc,linewidth=2)
	#plt.xscale("log")
	ylim_min = 30
	ylim_max=101
	plt.ylim(ylim_min,ylim_max)
	plt.xticks(fontweight=600,fontsize=10)
	plt.yticks(np.arange(ylim_min, ylim_max, 5.0),fontweight=600,fontsize=10)
	xlabel = xlabel[0] if xlabel else mode+' Feature Vector Length'
	ylabel = 'Accuracy (%)'
	plt.xlabel(xlabel,fontweight=800,fontsize=12)
	plt.ylabel(ylabel,fontweight=800,fontsize=12)
	plt.grid()
	plt.tight_layout()
	plt.savefig('./Results/'+mode+'_Accuraccy.png')

def Main():
	R = 1#radius of the circular pattern
	P = 8#number of points to sample on the circle
	classes = ['shine','sunrise','cloudy','rain']
	if not os.path.exists('./Results'):
		os.makedirs('./Results')

	image_dir = '../data'
	train_dir = image_dir+'/training'
	test_dir = image_dir+'/testing'
	training_images = os.listdir(train_dir)
	testing_images = os.listdir(test_dir)
	###############################################################################################################
	##########   Test the LBP Generator
	'''
	fname = ''
	TEXTURE = 'random'
	if fname:
		im = cv2.imread(fname)
	else:
		#im = np.ceil(np.random.rand(4,4)*10).astype(int)#random
		im = np.array([[5,4,2,4,2,2,4,0],[4,2,1,2,1,0,0,2],[2,4,4,0,4,0,2,4],[4,1,5,0,4,0,5,5],[0,4,4,5,0,0,3,2],[2,0,4,3,0,3,1,2],[5,1,0,0,5,4,2,3],[1,0,0,4,5,5,0,1]])#arbitrary
		#im = np.array([[5,0,5,0,5,0,5,0],[5,0,5,0,5,0,5,0],[5,0,5,0,5,0,5,0],[5,0,5,0,5,0,5,0],[5,0,5,0,5,0,5,0],[5,0,5,0,5,0,5,0],[5,0,5,0,5,0,5,0],[5,0,5,0,5,0,5,0]])#vertical
		#im = np.array([[5,5,5,5,5,5,5,5],[0,0,0,0,0,0,0,0],[5,5,5,5,5,5,5,5],[0,0,0,0,0,0,0,0],[5,5,5,5,5,5,5,5],[0,0,0,0,0,0,0,0],[5,5,5,5,5,5,5,5],[0,0,0,0,0,0,0,0]])#horizontal
		#im = np.array([[0,5,0,5,0,5,0,5],[5,0,5,0,5,0,5,0],[0,5,0,5,0,5,0,5],[5,0,5,0,5,0,5,0],[0,5,0,5,0,5,0,5],[5,0,5,0,5,0,5,0],[0,5,0,5,0,5,0,5],[5,0,5,0,5,0,5,0]])#checkerboard

	if len(im.shape)>2:
		#convert the image to grayscale
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	print('image = \n',im)
	LBP_HISTOGRAM = lbp.lbp(im,R=R,P=P,verbose=1)
	'''

	###############################################################################################################
	##########	 Generate LBP features for all images
	for R in range(1,11):
		LBP_SAVE_DIR = image_dir+'/LBP_HISTOGRAMS/R_'+str(R)
		if not os.path.exists(LBP_SAVE_DIR):
			os.makedirs(LBP_SAVE_DIR)
			os.makedirs(LBP_SAVE_DIR+'/training')
			os.makedirs(LBP_SAVE_DIR+'/testing')
		parameters = [R,P,0]
		Parallel(n_jobs=16)(delayed(prepare_LBP_histograms)(ix,training_images,train_dir,LBP_SAVE_DIR,'training',parameters) for ix in range(len(training_images)))
		Parallel(n_jobs=16)(delayed(prepare_LBP_histograms)(ix,testing_images,test_dir,LBP_SAVE_DIR,'testing',parameters) for ix in range(len(testing_images)))

	###############################################################################################################
	##########	 Generate GM features for all images
	#Prepare Gram-matrix Features from VGG Network
	# Load the model and the provided pretrained weights
	GM_SAVE_DIR = image_dir+'/GM_FEATURES'
	if not os.path.exists(GM_SAVE_DIR):
		os.makedirs(GM_SAVE_DIR)
		os.makedirs(GM_SAVE_DIR+'/training')
		os.makedirs(GM_SAVE_DIR+'/testing')
	VGG19_Model = vgg.VGG19()
	VGG19_Model.load_weights('vgg_normalized.pth')
	NI = len(training_images)
	for ix in range(0,NI):
		if training_images[ix][-4:] in ['.jpg','.JPG','.png','.PNG']:
			prepare_GM_features(training_images[ix],VGG19_Model,train_dir,GM_SAVE_DIR,'training',ix,NI)
	NI = len(testing_images)
	for ix in range(0,NI):
		if testing_images[ix][-4:] in ['.jpg','.JPG','.png','.PNG']:
			prepare_GM_features(testing_images[ix],VGG19_Model,test_dir,GM_SAVE_DIR,'testing',ix,NI)

	###############################################################################################################
	##########	 Generate AdaIN features for all images
	#Prepare AdaIN features from VGG Network
	# Load the model and the provided pretrained weights
	AdaIN_SAVE_DIR = image_dir+'/AdaIN_FEATURES'
	if not os.path.exists(AdaIN_SAVE_DIR):
		os.makedirs(AdaIN_SAVE_DIR)
		os.makedirs(AdaIN_SAVE_DIR+'/training')
		os.makedirs(AdaIN_SAVE_DIR+'/testing')
	VGG19_Model = vgg.VGG19()
	VGG19_Model.load_weights('vgg_normalized.pth')
	NI = len(training_images)
	for ix in range(0,NI):
		if training_images[ix][-4:] in ['.jpg','.JPG','.png','.PNG']:
			prepare_AdaIN_features(training_images[ix],VGG19_Model,train_dir,AdaIN_SAVE_DIR,'training',ix,NI)
	NI = len(testing_images)
	for ix in range(0,NI):
		if testing_images[ix][-4:] in ['.jpg','.JPG','.png','.PNG']:
			prepare_AdaIN_features(testing_images[ix],VGG19_Model,test_dir,AdaIN_SAVE_DIR,'testing',ix,NI)



	###############################################################################################################
	##########	 Plot the GM Feature matrix for a single image
	for image_id in range(1,100):
		try:
			X = np.load('../data/GM_FEATURES/training/'+training_images[image_id][:-4]+'.npz',allow_pickle=True)
			G = X['GM_FEATURE_VECTOR']
			G = np.reshape(G,[512,512])
			plot_GM(G,'GM_'+training_images[image_id][:-4])
		except:
			continue

	###############################################################################################################
	##########	 Training with GM Features
	if not os.path.exists('./Results/GM_Accuracies.npz'):
		GM_Accuracies = list()
		FEATURE_VECTOR_LENGTH = np.arange(1,512,2).tolist()
		FEATURE_VECTOR_LENGTH += np.arange(512,100000+512,1024).tolist()
		for kx in range(0,len(FEATURE_VECTOR_LENGTH)):
			#kx = FEATURE_VECTOR_LENGTH
			plot_cfm = True if FEATURE_VECTOR_LENGTH[kx]>=100000 else False
			[_,_,_,_,GM_ACCURACY] = train('../data/GM_FEATURES/training','../data/GM_FEATURES/testing','GM',classes,FEATURE_VECTOR_LENGTH[kx],verbose=1,plot_cfm=plot_cfm)
			GM_Accuracies.append(np.round(GM_ACCURACY*100,2))
		np.savez('./Results/GM_Accuracies.npz',GM_Accuracies=GM_Accuracies,FEATURE_VECTOR_LENGTH=FEATURE_VECTOR_LENGTH)
	plot_accuracy(['./Results/GM_Accuracies.npz'],'GM')


	###############################################################################################################
	##########	 Training with LBP Features
	LBP_Accuracies = list()
	for R in range(1,11):
		[X_train_LBP,Y_train_LBP,_,_,LBP_ACCURACY] = train('../data/LBP_HISTOGRAMS/R_'+str(R)+'/training','../data/LBP_HISTOGRAMS/R_'+str(R)+'/testing','LBP',classes,-1,verbose=1,plot_cfm=True,params=['_R_'+str(R)])
		LBP_Accuracies.append(np.round(LBP_ACCURACY*100,2))
		plot_LBP(X_train_LBP,Y_train_LBP,classes,R)
	np.savez('./Results/LBP_Accuracies.npz',LBP_Accuracies=LBP_Accuracies,FEATURE_VECTOR_LENGTH=[s for s in range(1,11)])
	plot_accuracy(['./Results/LBP_Accuracies.npz'],'LBP',xlabel=['LBP Radius of circular patter, R (#pixels)'])


	###############################################################################################################
	##########	 Training with AdaIN Features
	if not os.path.exists('./Results/AdaIN_Accuracies.npz'):
		AdaIN_Accuracies = list()
		FEATURE_VECTOR_LENGTH = np.arange(1,512*2+5,5).tolist()
		for kx in range(0,len(FEATURE_VECTOR_LENGTH)):
			#kx = FEATURE_VECTOR_LENGTH
			plot_cfm = True if FEATURE_VECTOR_LENGTH[kx]>=1024 else False
			[_,_,_,_,AdaIN_ACCURACY] = train('../data/AdaIN_FEATURES/training','../data/AdaIN_FEATURES/testing','AdaIN',classes,FEATURE_VECTOR_LENGTH[kx],verbose=1,plot_cfm=plot_cfm)
			AdaIN_Accuracies.append(np.round(AdaIN_ACCURACY*100,2))
		np.savez('./Results/AdaIN_Accuracies.npz',AdaIN_Accuracies=AdaIN_Accuracies,FEATURE_VECTOR_LENGTH=FEATURE_VECTOR_LENGTH)

	plot_accuracy(['./Results/AdaIN_Accuracies.npz'],'AdaIN')


Main()
