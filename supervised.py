import argparse
from itertools import cycle
from typing import Optional, Any

import keras
import numpy
import pandas
from keras.models import load_model
from sklearn.model_selection import train_test_split

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from norm import normalizeColumns

metadata = pandas.read_csv('metadata.csv', index_col='site_id')

print('Loading hdf5')
embeddings = pandas.read_hdf('embeddings.hdf','df')
print('Loaded .... normalizing data')

embeddings = normalizeColumns(metadata, embeddings)

# cull metadata rows not found in embeddings ....
metadata = metadata[metadata.index.isin(embeddings.index)]

metadata = metadata.sample(frac=1.0)                # shuffle

metadata = metadata[~metadata.disease_condition.isnull()]


#metadata = metadata.head(17)




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
	def __init__(self, dataFrame, batchSize = 16, stratify=True, loop=True, shuffle=True):
		self.df = dataFrame
		self.batchSize=batchSize

		self.labels = {'Mock':0, 'UV Inactivated SARS-CoV-2':0, 'Active SARS-CoV-2':1}

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
			features = embeddings.loc[index].values           # look up embedding vector for this index
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






















from keras import Input, Model
from keras.layers import Dropout, Dense
from keras.utils import plot_model


#from keras import backend as K



def buildModel(args):

	inp = Input(shape=(1024, ), name='embedding')
	x=inp

	x = Dense(args.fc1, activation='relu')(x)
	x = Dropout(args.dropout)(x)

	condition = Dense(1, activation='sigmoid', name='prediction')(x)

	model = Model(inputs=[inp], outputs=[condition])
	#model.compile(loss='mse', optimizer='adam')

	loss = 'binary_crossentropy'
	#loss = 'mse'
	model.compile(loss=loss, optimizer='adam', metrics=['acc'])
	#model.compile(loss='binary_crossentropy', optimizer=args.opt, metrics=['acc'])
	try: plot_model(model, to_file='model.png', show_shapes=True)
	except:
		print("NOTE: Couldn't render model.png")

	print(model.summary())

	return model



def train(args):
	model = buildModel(args)

	trainDF, testDF = train_test_split(metadata, test_size=0.2)

	# create dataset generators for keras
	tGen = DataGen(trainDF, batchSize=args.batchSize)
	steps = int(len(trainDF)/args.batchSize)

	vGen = DataGen(testDF, batchSize=512)       # doesnt effect training, but does effect performance!
	vsteps = int(len(testDF) / 512)

	# not using the "true" definition of a training epoch can be more convenient for realtime feedback of the model
	steps=100
	#vsteps=50

	callbacks=[]

	try:
		model.fit_generator(tGen, validation_data=vGen, validation_steps=vsteps, steps_per_epoch=steps, epochs=10000, callbacks=callbacks, verbose=1)
	except KeyboardInterrupt:  # allow aborting but enable saving
		pass
	finally:
		model.save('model.h5')

def finddrugs(args):

	model: keras.models.Model = load_model('model.h5')

	print(model.summary())

	#metadata = metadata.head()
	batchSize=4096
	gen = DataGen(metadata, loop=False, shuffle=False, stratify=False, batchSize=batchSize)
	steps = 1 + len(metadata)//batchSize

	predictions = model.predict_generator(gen, steps=steps)

	# remove the "batch overhang"
	print(len(metadata), predictions.shape)
	predictions = predictions[0: len(metadata)]

	metadata['prediction'] = predictions
	drugs = metadata[metadata.disease_condition=='Active SARS-CoV-2']
	drugs.sort_values('prediction', ascending=False, inplace=True)
	drugs.to_csv('drugPredictions.tsv', sep='\t')



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	#parser.add_argument('-cuda', action='store_true', help='Use CUDA accelerated tSNE. (never got this working)')
	parser.add_argument('-batchSize', type=int, default=16, help='')
	parser.add_argument('-dropout', type=float, default=0.1, help='')
	parser.add_argument('-fc1', type=int, default=4, help='')
	parser.add_argument('-train', action='store_true', help='')
	parser.add_argument('-finddrugs', action='store_true', help='')
	args = parser.parse_args()

	if args.train: train(args)

	if args.finddrugs: finddrugs(args)




