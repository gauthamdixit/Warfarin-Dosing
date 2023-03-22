import torch
import numpy as np
import torch.nn as nn
import pandas as pd 
from FeatureExtractor import Extractor
import random
import matplotlib.pyplot as plt
import torch.optim as optim

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork,self).__init__()
		self.input_layers = 5
		self.output_layers = 3
		self.dataframe = None
		self.extractor = Extractor()
		self.trueDoses = None
		self.dosageCategories = None
		self.splitPercent = 0


		self.net = nn.Sequential(
			nn.Linear(7,32),
			nn.ReLU(),
			nn.Linear(32,32),
			nn.ReLU(),
			nn.Linear(32,1))

			# 3 for categorical 1 for regressoin value


	def forward(self,x):
		x = self.net(x)
		return x

	def normalize(self,column):
		column = (column - column.min()) / (column.max() - column.min())
		return column

	def extractData(self,columnsToInclude,getCorr):
		self.extractor.setFile('../data/warfarin.csv')
		self.extractor.read_csv()
		
		weightMean = self.extractor.dataframe['Weight (kg)'].mean()
		heightMean = self.extractor.dataframe['Height (cm)'].mean()
		#self.extractor.dataframe = self.extractor.dataframe.dropna(subset=['Height (cm)','Weight (kg)'])
		self.splitPercent = self.extractor.splitPercent
		self.extractor.dataframe['Cyp2C9 genotypes'] = pd.factorize(self.extractor.dataframe['Cyp2C9 genotypes'])[0]
		self.extractor.dataframe['Race'] = pd.factorize(self.extractor.dataframe['Race'])[0]
		#self.extractor.dataframe['Gender'] = pd.factorize(self.extractor.dataframe['Gender'])[0]
		self.extractor.dataframe['VKORC1 genotype'] = pd.factorize(self.extractor.dataframe['VKORC1 genotype'])[0]
		#self.extractor.dataframe['Medications'] = pd.factorize(self.extractor.dataframe['Medications'])[0]
		self.extractor.dataframe['Height (cm)'].fillna(heightMean,inplace =True)
		self.extractor.dataframe['Amiodarone (Cordarone)'].fillna(0,inplace =True)
		self.extractor.dataframe['Weight (kg)'].fillna(weightMean,inplace =True)
		self.extractor.dataframe['Age'] = self.extractor.dataframe['Age'].str[:1].astype(int)

		self.extractor.dataframe['Height (cm)'] = self.normalize(self.extractor.dataframe['Height (cm)'])
		self.extractor.dataframe['Weight (kg)'] = self.normalize(self.extractor.dataframe['Weight (kg)'])
		#self.extractor.dataframe['Medications'] = self.normalize(self.extractor.dataframe['Medications'])
		#self.extractor.dataframe['Cyp2C9 genotypes'] = self.normalize(self.extractor.dataframe['Cyp2C9 genotypes'])
		#self.extractor.dataframe['Age'] = self.normalize(self.extractor.dataframe['Age'])
		#self.extractor.dataframe['VKORC1 genotype'] = self.normalize(self.extractor.dataframe['VKORC1 genotype'])
		#self.extractor.dataframe['Race'] = self.normalize(self.extractor.dataframe['Race'])

		if getCorr:
			self.dataframe = self.extractor.dataframe
			self.getCorrelation()

		self.extractor.dropData(columnsToInclude)
		self.trueDoses = self.extractor.extractGroundTruth()
		self.dataframe = self.extractor.dataframe

	def splitData(self,splitPercentage):
		splitIndex = int(self.dataframe.shape[0] * splitPercentage)
		train = np.array(self.dataframe.values[:splitIndex])
		test = np.array(self.dataframe.values[splitIndex:])
		return train,test,splitIndex

	def createDosageCategories(self):
		categories = []
		doses = self.trueDoses
		for i in range(len(doses)):
			if doses[i] < 21:
				categories.append(0)
			elif doses[i] >= 21 and doses[i] < 49:
				categories.append(1)
			else:
				categories.append(2)
		self.dosageCategories = categories

	def compareCategories(self,val1,val2):
		if val1 < 21 and val2 < 21:
			return True
		elif val1 >= 21 and val1 <49 and val2 >=21 and val2 < 49:
			return True
		elif val1 >= 49 and val2 >= 49:
			return True
		else:
			return False

net = NeuralNetwork()

#loss_func = nn.CrossEntropyLoss() #for categorical
loss_func = nn.MSELoss() # for regression
opt = optim.SGD(net.parameters(), lr = 0.001,momentum = 0.1)

columnsToInclude = ['Cyp2C9 genotypes','Age','Height (cm)','Weight (kg)', 'Race', 'VKORC1 genotype','Amiodarone (Cordarone)', 'Therapeutic Dose of Warfarin']
#bandit version below
#columnsToInclude = ['Age','Height (cm)','Weight (kg)','Cyp2C9 genotypes' ,'Race','Amiodarone (Cordarone)','VKORC1 genotype','enzyme inducer', 'Therapeutic Dose of Warfarin']

net.extractData(columnsToInclude,False)
train,test,index = net.splitData(net.splitPercent)
net.createDosageCategories()
lossList = []
epochs = []
prevLoss = 0
epoch = 0
dosesReal = net.trueDoses[index:]
validationAccuracy = []
validationIndex = []
while True:
	if epoch == 1500:
		break
	
	epochs.append(epoch)
	opt.zero_grad()
	trainTensor = torch.from_numpy(train).float()
	outputs = net(trainTensor)
	if epoch %10 == 0:
		print("epoch: ",epoch)
		testOutputs = net(torch.from_numpy(test).float())
		count = 0
		for i in range(len(testOutputs)):
			if net.compareCategories(testOutputs[i],dosesReal[i]):
				count+=1
		# if len(validationAccuracy) > 0:
		# 	if count/len(testOutputs) < validationAccuracy[-1]:
		# 		validationAccuracy.append(count/len(testOutputs))
		# 		validationIndex.append(epoch/1000)
		# 		break
		validationAccuracy.append(count/len(testOutputs))
		validationIndex.append(epoch)
	#print(net.trueDoses[:index])
	#loss = loss_func(outputs,torch.tensor(net.dosageCategories[:index]))
	loss = loss_func(outputs,torch.tensor(net.trueDoses[:index]).unsqueeze(1))
	lossList.append(loss.item())
	#print(loss)
	# if np.abs(loss.item() - prevLoss) < 0.0001:
	# 	print(loss.item())
	# 	print(prevLoss)
	# 	break
	prevLoss = loss.item()
	
	loss.backward()
	opt.step()
	epoch += 1

maxVal = np.argmax(validationAccuracy)
print("best epoch: ",validationIndex[maxVal])
print("best accuracy: ",validationAccuracy[maxVal])
outputs = net(torch.from_numpy(test).float())
#print(outputs)
predictions = torch.argmax(outputs,dim = 1)
#print(predictions)


count = 0
doses = net.dosageCategories[index:]

#for classification network
# for i in range(len(predictions)):
# 	if doses[i] == predictions[i]:
# 		count+=1
# print("accuracy: ", count/len(predictions))

for i in range(len(predictions)):
	if net.compareCategories(outputs[i],dosesReal[i]):
		count+=1
print("accuracy total: ", count,"/",len(predictions))
print("accuracy: ", count/len(predictions))

plt.plot(validationIndex,validationAccuracy,color = 'red')
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
plt.title("accuracy per epochs")
plt.savefig("validation.png")
plt.show()

plt.plot(epochs,lossList,color = 'red')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Loss per episode")
plt.savefig("Loss.png")
plt.show()