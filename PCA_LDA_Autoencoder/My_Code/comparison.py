import os
import gzip, pickle, pickletools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def main():
	save_dir = './Results'
	with open(save_dir+'/PCA/PCA_Accuracies.npy','rb') as f:
		PCA_Acc = np.load(f)
	with open(save_dir+'/LDA/LDA_Accuracies_original.npy','rb') as f:
		LDA_Acc = np.load(f)
	with open(save_dir+'/LDA/LDA_Accuracies_yu_yang.npy','rb') as f:
		LDAY_Acc = np.load(f)
	pretrained_weights = False
	if pretrained_weights:
		fname = save_dir+'/autoencoder_outputs/Autoencoder_Accuracies_pretrained.npy'
	else:
		fname = save_dir+'/autoencoder_outputs/Autoencoder_Accuracies_custom_trained.npy'
	with open(fname,'rb') as f:
		Autoenc_Acc = np.load(f)

	SMALL_SIZE = 10
	MEDIUM_SIZE = 12
	BIGGER_SIZE = 14
	LINE_WIDTH=3
	fig = plt.figure()
	plt.rc('font', size=SMALL_SIZE)
	plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	plt.plot(PCA_Acc,'-*',label="PCA", linewidth=LINE_WIDTH)
	plt.plot(LDA_Acc,'-o',label="LDA", linewidth=LINE_WIDTH)
	plt.plot(LDAY_Acc,'-d',label="LDA-Yu_Yang", linewidth=LINE_WIDTH)
	if pretrained_weights:
		plt.plot([3,8,16],Autoenc_Acc,'-s',label="Autoencoder (Pre-Trained)", linewidth=LINE_WIDTH)
		name = "pretrained"
	else:
		plt.plot(Autoenc_Acc,'-s',label="Autoencoder (Custom-Trained)", linewidth=LINE_WIDTH)
		name = "custom"
	plt.xticks(np.arange(0,len(PCA_Acc),2), [str(ix) for ix in range(1,len(PCA_Acc)+1,2)])
	plt.yticks(np.arange(20,101,10), [str(ix) for ix in range(20,101,10)])
	plt.legend()
	plt.grid()
	plt.xlabel('Number of Eigenvectors, P')
	plt.ylabel('Accuracy (%)')
	plt.tight_layout()
	plt.savefig(save_dir+"/Comparison_"+name+".pdf")
	plt.savefig(save_dir+"/Comparison_"+name+".png",dpi=600)
	plt.ylim([90,102])
	plt.yticks(np.arange(90,101,2), [str(ix) for ix in range(90,101,2)])
	plt.savefig(save_dir+"/Comparison_zoomed_"+name+".pdf")
	plt.savefig(save_dir+"/Comparison_zoomed_"+name+".png",dpi=600)
	plt.show()
main()
