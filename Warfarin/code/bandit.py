import numpy as np 
import pandas as pd 
from FeatureExtractor import Extractor
from linearModel import LinearModel
import random
import matplotlib.pyplot as plt

class LinearBandit:
	def __init__(self,extractor):
		self.c = 1
		self.d = 25#number of features
		self.arms = 3
		self.lmbda = 0.01
		self.dataframe = None
		self.extractor = extractor
		self.trueDoses = None
		self.A = [np.identity(self.d) for i in range(self.arms)]
		self.b = [np.zeros((self.d,1)) for i in range(self.arms)]
		self.theta = [np.zeros((self.d, 1)) for _ in range(self.arms)]
		self.dosageCategories = None
		self.regrets = [0]
		self.num_pulls = {0:0,1:0,2:0}
		self.splitPercent = 0
		self.t = 0

	def getCorrelation(self):
		target_col = 'Therapeutic Dose of Warfarin'
		corr_matrix = self.dataframe.corr()
		corr_with_target = corr_matrix[target_col]
		print("correlations: ",corr_with_target)

	def normalize(self,column):
		column = (column - column.min()) / (column.max() - column.min())
		return column

	def extractData(self,columnsToInclude,getCorr):
		self.extractor.setFile('../data/warfarin.csv')
		self.extractor.read_csv()

		self.splitPercent = self.extractor.splitPercent

		#self.extractor.dataframe = self.extractor.dataframe.dropna(subset=['Height (cm)','Weight (kg)'])
		weightMean = self.extractor.dataframe['Weight (kg)'].mean()
		heightMean = self.extractor.dataframe['Height (cm)'].mean()
		self.extractor.dataframe['Height (cm)'].fillna(heightMean,inplace =True)
		self.extractor.dataframe['Weight (kg)'].fillna(weightMean,inplace =True)
		self.extractor.dataframe['Carbamazepine (Tegretol)'].fillna(0,inplace =True)
		self.extractor.dataframe['Phenytoin (Dilantin)'].fillna(0,inplace =True)
		self.extractor.dataframe['Rifampin or Rifampicin'].fillna(0,inplace =True)
		self.extractor.dataframe['Amiodarone (Cordarone)'].fillna(0,inplace =True)
		self.extractor.dataframe['Race'].fillna("Unknown",inplace =True)
		self.extractor.dataframe['Cyp2C9 genotypes'].fillna("Unknown",inplace =True)
		self.extractor.dataframe['VKORC1 genotype'].fillna("Unknown",inplace =True)
		#self.extractor.dataframe['Gender'].fillna("Unknown",inplace =True)

		self.extractor.dataframe['Age'] = self.extractor.dataframe['Age'].str[:1].astype(int)

		#self.extractor.dataframe['Amiodarone (Cordarone)'] = self.normalize(self.extractor.dataframe['Amiodarone (Cordarone)'])
		#self.extractor.dataframe['Height (cm)'] = self.normalize(self.extractor.dataframe['Height (cm)'])
		#self.extractor.dataframe['Weight (kg)'] = self.normalize(self.extractor.dataframe['Weight (kg)'])
		#self.extractor.dataframe['Age'] = self.normalize(self.extractor.dataframe['Age'])
		

		#CategoricalColumns = ['Cyp2C9 genotypes','VKORC1 genotype','Race','Gender']
		#CategoricalColumns = ['Race','Gender']
		CategoricalColumns = ['Cyp2C9 genotypes','VKORC1 genotype','Race']
		
		carb = self.extractor.dataframe['Carbamazepine (Tegretol)'].tolist()
		phen = self.extractor.dataframe['Phenytoin (Dilantin)'].tolist()
		rif = self.extractor.dataframe['Rifampin or Rifampicin'].tolist()

		self.extractor.dataframe['enzyme inducer'] = [1 if carb[i] == 1 or phen[i] == 1 or rif[i] == 1 else 0 for i in range(len(rif))]
		# if getCorr:
		# 	self.dataframe = self.extractor.dataframe
		# 	self.getCorrelation()

		self.extractor.dropData(columnsToInclude)
		one_hot_df = pd.get_dummies(self.extractor.dataframe, columns=CategoricalColumns)
		self.trueDoses = self.extractor.extractGroundTruth()
		self.dataframe = one_hot_df
		self.dataframe = self.dataframe.drop('Race_White',axis =1)	
		#self.dataframe = (self.dataframe - self.dataframe.mean()) / self.dataframe.std()
		
		#print(self.dataframe)

	def constructBalancedTrainSet(self,train,r,minCount):
		trainData = []
		r_data = []
		countDict = {}
		for i in range(len(r)):
			st = ', '.join(str(r[i]))
			if st not in countDict:
				countDict[st] = 0
			if countDict[st] < minCount:
				trainData.append(train[i])
				r_data.append(r[i])
				countDict[st] += 1
		print(countDict)
		return np.array(trainData),np.array(r_data)


	# def constructRewardMatrix(self):
	# 	rewardMatrix = []
	# 	dosesList = self.trueDoses
	# 	lowCount = 0
	# 	midCount = 0
	# 	highCount = 0
	# 	for i in range(len(dosesList)):
	# 		if dosesList[i] < 21:
	# 			lowCount+=1
	# 			rewardMatrix.append([0,-0.75,-3])
	# 		elif dosesList[i] >= 21 and dosesList[i] < 49:
	# 			midCount+=1
	# 			rewardMatrix.append([-0.75,0,-0.75])
	# 		else:
	# 			highCount+=1
	# 			rewardMatrix.append([-1.5,-0.75,0])
	# 	print([lowCount,midCount,highCount])
	# 	minCount = np.min([lowCount,midCount,highCount])
	# 	return rewardMatrix,minCount
	def constructRewardMatrix(self):
		rewardMatrix = []
		dosesList = self.trueDoses
		lowCount = 0
		midCount = 0
		highCount = 0
		for i in range(len(dosesList)):
			if dosesList[i] < 21:
				lowCount+=1
				rewardMatrix.append([0,-1,-20])
			elif dosesList[i] >= 21 and dosesList[i] < 49:
				midCount+=1
				rewardMatrix.append([-1,0,-1])
			else:
				highCount+=1
				rewardMatrix.append([-10,-1,0])
		minCount = np.min([lowCount,midCount,highCount])
		return rewardMatrix,minCount

	def splitData(self,splitPercentage):
		splitIndex = int(self.dataframe.shape[0] * splitPercentage)
		train = np.array(self.dataframe.values[:splitIndex])
		test = np.array(self.dataframe.values[splitIndex:])
		return train,test,splitIndex
	
	def predict(self, x):
		p = np.zeros(self.arms)
		#x = np.reshape(x, (self.d, 1))
		for i in range(self.arms):
			theta = self.theta[i]
			if self.num_pulls[i] == 0:
				alpha = np.inf
			else:
				alpha = self.c * np.sqrt(np.log(self.t+1) / self.num_pulls[i])
			p[i] = np.dot(theta.T,x) + alpha*np.sqrt(np.dot(x.T,np.dot(self.A[i],x)))
		action = np.argmax(p)
		self.num_pulls[action] += 1
		return action

	def update(self,a, x, y):
		#x = np.reshape(x, (self.d, 1))
		self.A[a] += np.dot(x,x.T)
		self.b[a] += y * x[:,np.newaxis]
		self.theta[a] = np.linalg.solve(self.A[a]+ self.lmbda * np.eye(self.d) ,self.b[a])#np.linalg.inv(self.A[a] + self.lmbda * np.eye(self.d)).dot(self.b[a])
		self.t+=1
		
	def train(self,X,rewards):
		regret = 0
		count = 0
		fractionOfWrong = [0]
		for i in range(len(X)):
			action = self.predict(X[i])
			reward = rewards[i][action]
			if reward == 0:
				count+=1
			regret += (0-reward)
			self.regrets.append(regret)
			fractionOfWrong.append(1-(count/(i+1)))
			self.update(action,X[i],reward)
		return regret,fractionOfWrong


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
		self.dosageCategories = np.array(categories)

def testModel(alpha):
	extractor = Extractor()
	bandit = LinearBandit(extractor)
	bandit.c = alpha
	#columnsToInclude = ['Age','Height (cm)','Weight (kg)','Gender' ,'Race','Amiodarone (Cordarone)','VKORC1 genotype','Cyp2C9 genotypes', 'Therapeutic Dose of Warfarin']
	columnsToInclude = ['Cyp2C9 genotypes','enzyme inducer','VKORC1 genotype','Age','Height (cm)','Weight (kg)','Race','Amiodarone (Cordarone)','Therapeutic Dose of Warfarin']

	bandit.extractData(columnsToInclude,False)

	train = np.array(bandit.dataframe.values)
	bandit.createDosageCategories()
	r,minCount = bandit.constructRewardMatrix()
	#train using bandit algorithm

	i = 0
	#train,r = bandit.constructBalancedTrainSet(train,r,minCount)
	r = np.array(r)
	prevRegret = 0
	permutation_indices = np.random.permutation(len(train)).astype(int)
	trainPerm = train[permutation_indices]
	rPerm = r[permutation_indices,:]
	reg,accuracy = bandit.train(trainPerm,rPerm)
	prevRegret = reg
	regrets = bandit.regrets

	return regrets,accuracy

regrets = []
accuracies = []

for i in range(20):
	print("trials run: ",i)
	regret,accuracy = testModel(300)
	regrets.append(regret)
	accuracies.append(accuracy)
x = [i for i in range(len(regrets[0]))]

np_regrets = np.array(regrets)
np_accuracy = np.array(accuracies)

mean_regrets = np.mean(np_regrets,axis = 0)
std_regrets = np.std(np_regrets,axis = 0)
mean_accuracy = np.mean(np_accuracy,axis = 0)

# Calculate upper and lower bounds of the 95% confidence interval
lower_bound = mean_regrets - 1.96 * std_regrets / np.sqrt(len(regrets))
upper_bound = mean_regrets + 1.96 * std_regrets / np.sqrt(len(regrets))
print("max accuracy: ", 1-np.sort(mean_accuracy)[1])
print("average accuracy: ", 1-np.mean(mean_accuracy))

extractor = Extractor()
model = LinearModel(extractor)
columnsToInclude = ['Age','Height (cm)','Weight (kg)', 'Race', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin','Amiodarone (Cordarone)','Therapeutic Dose of Warfarin']
#columnsToInclude = ['Age']
model.extractData(columnsToInclude)
model.createDosageCategories()
reg1 = model.testFixedDose(35)
reg2,total = model.testLinearModel()
reg1.insert(0, 0)
reg2.insert(0, 0)
plt.plot(x,mean_regrets,"-r", label="bandit")
plt.fill_between(x, lower_bound, upper_bound, alpha=0.2, color='red')
plt.plot(x,reg1,"-b", label="Fixed Dose")
plt.plot(x,reg2,"-g", label="clinical dosing algorithm")
plt.legend(loc="upper left")
plt.xlabel('patients seen')
plt.ylabel('cumultive regret')
plt.title("Regret per patient seen")
plt.savefig("regret.png")
plt.show()

plt.plot(x,mean_accuracy,"-b", label="Clinical Dosing Algorithm")
plt.xlabel('patients seen')
plt.ylabel('fraction incorrect')
plt.title("percent incorrect per patient seen")
plt.savefig("percentIncorrBandit.png")
plt.show()
		

