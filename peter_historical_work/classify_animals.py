import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

cudnn.benchmark = True


###############################################################
# formatting data
###############################################################

# data transformations
data_transforms = {
	'train': transforms.Compose([
		transforms.CenterCrop((1400,2048)),
		transforms.Resize((350,512)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
	]),
	'test': transforms.Compose([
		transforms.CenterCrop((1200,2048)),
		transforms.Resize((300,512)),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
	]),
}

data_dir = './animals_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','test']}
class_names = image_datasets['train'].classes

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


###############################################################
# Model Training
###############################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs=8):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print(f'Epoch {epoch}/{num_epochs-1}')
		print('-'*10)

		# Each epoch has a training and validation phase
		for phase in ['train','test']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward, track history only if in train phase
				with torch.set_grad_enabled(phase=='train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize on if in train phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			
			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

			# deep copy the model
			if phase == 'test' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()
	
	time_elapsed = time.time() - since
	print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
	print(f'Best test Acc: {best_acc:4f}')

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model


###############################################################
# model testing
###############################################################

def test_model(model,dataloaders):
	CM = 0
	model.eval()
	with torch.no_grad():
		for images, labels in dataloaders['test']:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			preds = torch.argmax(outputs.data,1)
			CM+=confusion_matrix(labels.cpu(),preds.cpu(),labels=[0,1])

		tn=CM[0][0]
		tp=CM[1][1]
		fp=CM[0][1]
		fn=CM[1][0]
		acc=np.sum(np.diag(CM)/np.sum(CM))
		sensitivity=tp/(tp+fn)
		precision=tp/(tp+fp)

		print('===============================================================')
		print('Test Accuracy (mean): %f %%' % (100*acc))
		print('Confusion Matrix: ', CM)
		print('- Sensitivity: ',(tp/(tp+fn))*100)
		print('- Specificity: ',(tn/(tn+fp))*100)
		print('- Precision: ',(tp/(tp+fp))*100)
		print('- NPV: ',(tn/(tn+fn))*100)
		print('- F1: ',((2*sensitivity*precision)/(sensitivity+precision))*100)
		print('===============================================================')
		print('Model Details:')
		print(model)
		print('===============================================================')
		print('Dataset Sizes: ', dataset_sizes)
		print('===============================================================')
		print('Predictions: ')
		print(preds.cpu())
		print('===============================================================')
		print('Labels: ')
		print(labels.cpu())
		print('===============================================================')



###############################################################
# prediction visualization
###############################################################

def imshow(inp,title=None):
	"""Imshow for Tensor"""
	#fig = plt.figure(figsize=(16,12))
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485,0.456,0.406])
	std = np.array([0.229,0.224,0.225])
	inp = std * inp + mean
	inp = np.clip(inp,0,1)
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)

def visualize_model(model, num_images=50):
	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure(figsize=(24,12))

	with torch.no_grad():
		for i, (inputs, labels) in enumerate(dataloaders['test']):
			inputs = inputs.to(device)
			labels = labels.to(device)
			
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			for j in range(inputs.size()[0]):
				images_so_far += 1
				ax = plt.subplot(num_images//10, 10, images_so_far)
				ax.axis('off')
				ax.set_title(f'predicted: {class_names[preds[j]]}')
				imshow(inputs.cpu().data[j])

				if images_so_far == num_images:
					model.train(mode=was_training)
					return
		model.train(mode=was_training)

###############################################################
# Loading pretrained model
###############################################################

# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# # set output sample to 2 (only 2 classes)
# model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
# change number of nodes in last two layers
model_ft.classifier[4] = nn.Linear(4096,1024)
model_ft.classifier[6] = nn.Linear(1024,2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)


# ###############################################################
# # train and test
# ###############################################################

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=8)
test_model(model_ft,dataloaders)
# visualize_model(model_ft)
# plt.savefig('prediction.png')
