import numpy as np
import pandas as pd

class Extractor:
	def __init__(self,filename = None):
		self.filename = filename 
		self.dataframe = None
		self.splitPercent = 0.8

	def setFile(self,name):
		self.filename = name

	def read_csv(self):
		try:
			df = pd.read_csv(self.filename)
			self.dataframe	= df
			#print(self.dataframe.columns.tolist())
		except FileNotFoundError:
			print(f"Error: file '{self.filename}' not found.")
			return None

	def dropData(self,columnsToInclude):

		columnsToDrop = []
		columns = self.dataframe.columns.tolist()

		for feature in columns:
			if feature not in columnsToInclude:
				columnsToDrop.append(feature)
		
		self.dataframe = self.dataframe.drop(columns = columnsToDrop,axis = 1)
		new_columns = self.dataframe.columns.tolist()
		self.dataframe.dropna(subset=['Therapeutic Dose of Warfarin'], inplace=True)

	def extractGroundTruth(self):
		answerVector = self.dataframe['Therapeutic Dose of Warfarin'].tolist()
		#print("truth values: ",answerVector)
		self.dataframe = self.dataframe.drop(columns = ['Therapeutic Dose of Warfarin'])
		return answerVector

  # do in this order:
  #create class -> setFile -> read_csv -> extractGroundTruth -> dropData
