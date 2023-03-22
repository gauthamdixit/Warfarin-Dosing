from FeatureExtractor import Extractor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearModel:
	def __init__(self,extractor):
		self.extractor = extractor
		self.dataframe = None
		self.trueDoses = None
		self.dosageCategories = []
		self.fixedDoseCount = 0
		self.splitPercent = 0	

	def extractData(self,columnsToInclude):
		self.extractor.setFile('../data/warfarin.csv')
		self.extractor.read_csv()
		self.extractor.dropData(columnsToInclude)
		self.trueDoses = self.extractor.extractGroundTruth()
		self.dataframe = self.extractor.dataframe
		self.dataframe['Age'] = self.dataframe['Age'].str[:1].astype(int)
		heightmean = self.dataframe['Height (cm)'].mean()
		weightmean = self.dataframe['Weight (kg)'].mean()
		self.dataframe['Height (cm)'].fillna(heightmean,inplace =True)
		self.dataframe['Weight (kg)'].fillna(weightmean,inplace =True)
		self.dataframe["Race"].fillna("Unknown",inplace=True)
		self.dataframe['Carbamazepine (Tegretol)'].fillna(0,inplace =True)
		self.dataframe['Phenytoin (Dilantin)'].fillna(0,inplace =True)
		self.dataframe['Rifampin or Rifampicin'].fillna(0,inplace =True)
		self.dataframe['Amiodarone (Cordarone)'].fillna(0,inplace =True)
		self.splitPercent = self.extractor.splitPercent

	def createDosageCategories(self):
		categories = []
		for i in range(len(self.trueDoses)):
			if self.trueDoses[i] < 21:
				categories.append(0)
			elif self.trueDoses[i] >= 21 and self.trueDoses[i] < 49:
				categories.append(1)
			else:
				categories.append(2)
		self.dosageCategories = np.array(categories)

	def testFixedDose(self,dosage):
		train,test,index = self.splitData(self.splitPercent)
		cumaltiveReg = 0
		count = 0
		regrets = []
		pSeen = []
		tSet = self.dosageCategories
		for i in range(len(tSet)):
			if tSet[i] == 1:
				count += 1
			else:
				cumaltiveReg += 1
			regrets.append(cumaltiveReg)
			pSeen.append(i)
		print("fixed dose accuracy: ", count/len(tSet))

		return regrets

	def splitData(self,splitPercentage):
		splitIndex = int(self.dataframe.shape[0] * splitPercentage)
		train = np.array(self.dataframe.values[:splitIndex])
		test = np.array(self.dataframe.values[splitIndex:])
		return train,test,splitIndex



		#finish this
	def testLinearModel(self):
		dosages = []
		correct = 0 
		predictions = []
		train,test,index = self.splitData(self.splitPercent)
		for i in range(0,len(self.dosageCategories)):
			#need to clean up age
			race = self.dataframe["Race"].tolist()[i]
			asianValue = 0
			blackValue = 0
			unknown = 0
			if race == "Asian":
				asianValue = 1
			elif race == "Black or African American":
				blackValue = 1
			elif race == "Unknown":
				unknown = 1
			enzymeInducer = 0
			amiodarone = 0
			if self.dataframe['Carbamazepine (Tegretol)'].tolist()[i] == 1 or self.dataframe['Phenytoin (Dilantin)'].tolist()[i] == 1 or self.dataframe['Rifampin or Rifampicin'].tolist()[i] == 1:
				enzymeInducer = 1
			if self.dataframe['Amiodarone (Cordarone)'].tolist()[i] == 1:
				amiodarone = 1
			

			dose = 4.0376 - 0.2546*(self.dataframe['Age'].tolist()[i]) + \
			0.0118*self.dataframe['Height (cm)'].tolist()[i] + \
			0.0134*self.dataframe['Weight (kg)'].tolist()[i] - \
			0.6752*asianValue + \
			0.4060*blackValue + \
			0.0443*unknown + \
			1.2799*enzymeInducer - \
			0.5695*amiodarone
			dose = (dose**2)
			result = 0
			if dose < 21:
				predictions.append(0)
			elif dose >= 21 and dose < 49:
				predictions.append(1)
			else:
				predictions.append(2)
		predictions = np.array(predictions)
		cumaltiveReg = 0
		regrets = []
		pSeen = []
		correctPercent = []
		for i in range(len(predictions)):
			if predictions[i] == self.dosageCategories[i]:
				correct += 1
			else:
				if predictions[i] == 0:
					if self.dosageCategories[i] == 1:
						cumaltiveReg+= 1
					elif self.dosageCategories[i] == 2:
						cumaltiveReg+= 10
				elif predictions[i] == 2:
					if self.dosageCategories[i] == 0:
						cumaltiveReg += 20
					elif self.dosageCategories[i] == 1:
						cumaltiveReg += 1
				else:
					cumaltiveReg += 1	
			regrets.append(cumaltiveReg)
			pSeen.append(i+1)
			correctPercent.append(correct/(i+1))


		total = len(test)+len(train)
		print("linear model results: ",correct,"/",total)
		print("linear model accuracy: ",correct/total) 
		return regrets,pSeen




#['Race','Age','Weight (kg)','Height (cm)','Medications']


# extractor = Extractor()
# model = LinearModel(extractor)
# columnsToInclude = ['Age','Height (cm)','Weight (kg)', 'Race', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin','Amiodarone (Cordarone)','Therapeutic Dose of Warfarin']
# #columnsToInclude = ['Age']
# model.extractData(columnsToInclude)
# model.createDosageCategories()
# reg1 = model.testFixedDose(35)
# reg2,total = model.testLinearModel()

# plt.plot(total,reg1,"-r", label="Fixed Dose")
# plt.plot(total,reg2,"-b", label="Clinical Dosing Algorithm")
# plt.legend(loc="upper left")
# #plt.fill_between(x, lower_bound, upper_bound, alpha=0.2, color='red')
# plt.xlabel('patients seen')
# plt.ylabel('cumaltive regret')
# plt.savefig("baseline_regret_linear.png")
# plt.show()		
