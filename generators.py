from itertools import cycle

import numpy
from keras import backend as K


class DataframeGenerator():
	def __init__(self, dataframe, loop=True, shuffle=True):
		self.loop = loop
		self.shuffle = shuffle
		assert len(dataframe) > 0, 'Dataframe is empty!'
		self.df = dataframe
		self.gen = self.start()

	def start(self):
		if self.shuffle: self.df = self.df.sample(frac=1.0)
		self.gen = self.df.iterrows()
		return self.gen

	def __next__(self):
		try: return next(self.gen)
		except StopIteration:
			if self.loop:
				self.start()
				return next(self.gen)
			else: raise StopIteration


class DataGen():
	def __init__(self, dataFrame, embeddings, batchSize = 16, stratify=True, loop=True, shuffle=True):
		self.df = dataFrame
		self.batchSize=batchSize

		self.labels = {'Active SARS-CoV-2':0, 'Mock':1, 'UV Inactivated SARS-CoV-2':1, }
		self.embeddings = embeddings

		# setup stratified subsampling
		if stratify:
			uniqLabels = dataFrame.disease_condition.unique()
			dataFrameParts = []
			for value in uniqLabels:
				subDF = dataFrame[dataFrame.disease_condition == value]
				dataFrameParts.append(subDF)
		else : dataFrameParts = [dataFrame]

		# turn these into generators
		generators = [DataframeGenerator(df, loop=loop, shuffle=shuffle) for df in dataFrameParts]

		# consruct an infinite generator that loops between these partitions
		self.generators = cycle(generators)


	def nextInstance(self):
		# get next generator in the stratified subsampling list
		gen = next(self.generators)
		# get row from this selected generator
		try:
			index, row = next(gen)
			label = row.disease_condition
			label = self.labels[label]
			features = self.embeddings.loc[index].values           # look up embedding vector for this index
			features = features.astype(K.floatx())
		except StopIteration:                                 # for evaluation of the model, we dont loop infinitely
			features = numpy.zeros(1024)
			label = -1          # a skip code for later

		return features, label


	def nextBatch(self):
		featureBatch, labelBatch = [], []
		for i in range(0, self.batchSize):
			instance = self.nextInstance()
			features, label = instance
			featureBatch.append(features)
			labelBatch.append([label])
		featureBatch = numpy.array(featureBatch)
		labelBatch = numpy.array(labelBatch)#.T
		return featureBatch, labelBatch

	def __next__(self):             # keras expects batches
		#return next(self.bgen)
		return self.nextBatch()