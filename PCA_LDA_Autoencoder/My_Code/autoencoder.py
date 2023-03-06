import os

import numpy as np
import torch
from torch import nn, optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DataBuilder(Dataset):
	def __init__(self, path):
		self.path = path
		self.image_list = [f for f in os.listdir(path) if f.endswith('.png')]
		self.label_list = [int(f.split('_')[0]) for f in self.image_list]
		self.len = len(self.image_list)
		self.aug = transforms.Compose([
			transforms.Resize((64, 64)),
			transforms.ToTensor(),
		])

	def __getitem__(self, index):
		fn = os.path.join(self.path, self.image_list[index])
		x = Image.open(fn).convert('RGB')
		x = self.aug(x)
		return {'x': x, 'y': self.label_list[index]}

	def __len__(self):
		return self.len
class Autoencoder(nn.Module):

	def __init__(self, encoded_space_dim):
		super().__init__()
		self.encoded_space_dim = encoded_space_dim
		### Convolutional section
		self.encoder_cnn = nn.Sequential(
			nn.Conv2d(3, 8, 3, stride=2, padding=1),
			nn.LeakyReLU(True),
			nn.Conv2d(8, 16, 3, stride=2, padding=1),
			nn.LeakyReLU(True),
			nn.Conv2d(16, 32, 3, stride=2, padding=1),
			nn.LeakyReLU(True),
			nn.Conv2d(32, 64, 3, stride=2, padding=1),
			nn.LeakyReLU(True)
		)
		### Flatten layer
		self.flatten = nn.Flatten(start_dim=1)
		### Linear section
		self.encoder_lin = nn.Sequential(
			nn.Linear(4 * 4 * 64, 128),
			nn.LeakyReLU(True),
			nn.Linear(128, encoded_space_dim * 2)
		)
		self.decoder_lin = nn.Sequential(
			nn.Linear(encoded_space_dim, 128),
			nn.LeakyReLU(True),
			nn.Linear(128, 4 * 4 * 64),
			nn.LeakyReLU(True)
		)
		self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 4, 4))
		self.decoder_conv = nn.Sequential(
			nn.ConvTranspose2d(64, 32, 3, stride=2,padding=1, output_padding=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(True),
			nn.ConvTranspose2d(32, 16, 3, stride=2,padding=1, output_padding=1),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(True),
			nn.ConvTranspose2d(16, 8, 3, stride=2,padding=1, output_padding=1),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(True),
			nn.ConvTranspose2d(8, 3, 3, stride=2,padding=1, output_padding=1)
		)

	def encode(self, x):
		x = self.encoder_cnn(x)
		x = self.flatten(x)
		x = self.encoder_lin(x)
		mu, logvar = x[:, :self.encoded_space_dim], x[:, self.encoded_space_dim:]
		return mu, logvar

	def decode(self, z):
		x = self.decoder_lin(z)
		x = self.unflatten(x)
		x = self.decoder_conv(x)
		x = torch.sigmoid(x)
		return x

	@staticmethod
	def reparameterize(mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = Variable(std.data.new(std.size()).normal_())
		return eps.mul(std).add_(mu)
class VaeLoss(nn.Module):
	def __init__(self):
		super(VaeLoss, self).__init__()
		self.mse_loss = nn.MSELoss(reduction="sum")

	def forward(self, xhat, x, mu, logvar):
		loss_MSE = self.mse_loss(xhat, x)
		loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return loss_MSE + loss_KLD
def train(epoch,epochs,p):
	model.train()
	train_loss = 0

	for batch_idx, data in enumerate(trainloader):
		optimizer.zero_grad()
		mu, logvar = model.encode(data['x'])
		z = model.reparameterize(mu, logvar)
		xhat = model.decode(z)
		loss = vae_loss(xhat, data['x'], mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()

	print('====> P: {} Epoch: {}/{} Average loss: {:.4f}'.format(p,epoch,epochs, train_loss / len(trainloader.dataset)))
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
##################################
# Change these
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)
training = False
pretrained_weights = False


TRAIN_DATA_PATH =  '../FaceRecognition/train'
EVAL_DATA_PATH = '../FaceRecognition/test'
OUT_PATH = './Results/autoencoder_outputs'
if pretrained_weights:
	WEIGHT_DIR = './autoencoder_weights'
	P = [3,8,16]
else:
	WEIGHT_DIR = './Results/autoencoder_outputs'
	P = np.arange(1,26,1)
num_classes = 30
train_image_per_class = 21
test_image_per_class = 21
classifier = "KNN"#options = KNN,L2
params = [1]#KNN neighbour
if not os.path.exists(OUT_PATH):
	os.makedirs(OUT_PATH)
##################################
Accuracies = list()
for p in P:
	LOAD_PATH = WEIGHT_DIR+'/model_'+str(p)+'.pt'
	model = Autoencoder(p)

	if training:
		epochs = 100
		log_interval = 1
		trainloader = DataLoader(
			dataset=DataBuilder(TRAIN_DATA_PATH),
			batch_size=12,
			shuffle=True,
		)
		optimizer = optim.Adam(model.parameters(), lr=1e-3)
		vae_loss = VaeLoss()
		for epoch in range(1, epochs + 1):
			train(epoch,epochs,p)
		torch.save(model.state_dict(), os.path.join(OUT_PATH, f'model_{p}.pt'))
	else:
		trainloader = DataLoader(
			dataset=DataBuilder(TRAIN_DATA_PATH),
			batch_size=1,
		)

		model.load_state_dict(torch.load(LOAD_PATH))
		model.eval()
		model.to(device)

		X_train = []
		Y_train = []
		for batch_idx, data in enumerate(trainloader):
			mu, logvar = model.encode(data['x'])
			z = mu.detach().cpu().numpy().flatten()
			X_train.append(z)
			Y_train.append(data['y'].item())
		X_train = np.stack(X_train)
		Y_train = np.array(Y_train)

		testloader = DataLoader(
			dataset=DataBuilder(EVAL_DATA_PATH),
			batch_size=1,
		)
		X_test, Y_test = [], []
		for batch_idx, data in enumerate(testloader):
			mu, logvar = model.encode(data['x'])
			z = mu.detach().cpu().numpy().flatten()
			X_test.append(z)
			Y_test.append(data['y'].item())
		X_test = np.stack(X_test)
		Y_test = np.array(Y_test)

		distance = np.zeros((X_test.shape[0],X_train.shape[0]))

		for ix in range(0,X_test.shape[0]):
			distance[ix,:] = np.sqrt(np.sum(np.square(X_train-X_test[ix,:]),axis=1)).T
		accuracy = calculate_accuracy(Y_test,distance,test_image_per_class,num_classes,method=classifier,params=params)
		print("P = ",p,", Accuracy = ",accuracy,'%')
		Accuracies.append(accuracy)
	if pretrained_weights:
		fname = OUT_PATH+'/Autoencoder_Accuracies_pretrained.npy'
	else:
		fname = OUT_PATH+'/Autoencoder_Accuracies_custom_trained.npy'
	with open(fname, 'wb') as f:
		np.save(f,Accuracies)
