from keras import backend as K

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


#metadata = metadata.head(50000)                 # for testing smaller quantities




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

		self.labels = {'Active SARS-CoV-2':0, 'Mock':1, 'UV Inactivated SARS-CoV-2':1, }

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






















from keras import Input, Model
from keras.layers import Dropout, Dense
from keras.utils import plot_model


#from keras import backend as K



def buildModel(args):

	inp = Input(shape=(1024, ), name='embedding')
	x=inp

	x = Dense(args.fc1, activation='tanh')(x)
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
	vsteps=100

	callbacks=[]

	try:
		model.fit_generator(tGen, validation_data=vGen, validation_steps=vsteps, steps_per_epoch=steps, epochs=10000, callbacks=callbacks, verbose=1)
	except KeyboardInterrupt:  # allow aborting but enable saving
		pass
	finally:
		model.save('model.h5')

def finddrugs(args):
	'''
	Basic idea :

	we trained a model to predict drug treatment (0) or Mock/UV inactivated (1) samples

	Now we loop over the drugs to find which ones LOOK LIKE Mock/UV inactivated samples .... ;)

	'''


	model: keras.models.Model = load_model('model.h5')

	print(model.summary())

	#metadata = metadata.head()
	batchSize=4096*4
	gen = DataGen(metadata, loop=False, shuffle=False, stratify=False, batchSize=batchSize)
	steps = 1 + len(metadata)//batchSize

	predictions = model.predict_generator(gen, steps=steps)

	'''
	print('Generator produced this predictions array :', predictions.shape, predictions.dtype)
	#predictions = predictions.astype(float64)
	predictions.tofile('predictions.py')
	predictions = numpy.fromfile('predictions.py', dtype=K.floatx())
	print('Loaded this predictions array :', predictions.shape)
	'''

	# remove the "batch overhang"
	print(len(metadata), predictions.shape)
	predictions = predictions[0: len(metadata)]

	metadata['prediction'] = predictions
	drugs = metadata[metadata.disease_condition=='Active SARS-CoV-2']
	drugs.sort_values('prediction', ascending=False, inplace=True)
	drugs.to_csv('drugPredictions.tsv', sep='\t')

	'''
	print('predictions counts : ')
	print(metadata.prediction.value_counts())
	weird = metadata[metadata.prediction == 0.6428735256195068]
	print(weird)
	'''

	import matplotlib.pyplot as plt
	# An "interface" to matplotlib.axes.Axes.hist() method
	n, bins, patches = plt.hist(x=predictions, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.title('Output probabilities that drug-treated state is mistaken for Mock/Inactive')
	plt.text(23, 45, r'$\mu=15, b=3$')
	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=numpy.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


	plt.savefig('drughist.png', dpi=300)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	#parser.add_argument('-cuda', action='store_true', help='Use CUDA accelerated tSNE. (never got this working)')
	parser.add_argument('-batchSize', type=int, default=16, help='')
	parser.add_argument('-dropout', type=float, default=0.1, help='')
	parser.add_argument('-fc1', type=int, default=8, help='')
	parser.add_argument('-train', action='store_true', help='')
	parser.add_argument('-finddrugs', action='store_true', help='')
	args = parser.parse_args()

	if args.train: train(args)

	if args.finddrugs: finddrugs(args)




