import argparse
import json
import random

import math
import pandas
from keras import Input, Model
from keras.layers import Dense, Dropout, Embedding, Reshape, Flatten
from keras.utils import plot_model

import selfies
import numpy as np

import selfies as sf




#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from norm import normalizeColumns

'''
metadata = pandas.read_csv('metadata-big.csv', index_col='site_id')     # this version has SMILES strings

#print('Loading hdf5')
#embeddings = pandas.read_hdf('embeddings.hdf','df')
#print('Loaded .... normalizing data')
#embeddings = normalizeColumns(metadata, embeddings)

# cull metadata rows not found in embeddings ....
#metadata = metadata[metadata.index.isin(embeddings.index)]
metadata = metadata.sample(frac=1.0)                # shuffle
metadata = metadata[~metadata.disease_condition.isnull()]
'''


def convertSelfies():
	metadata = pandas.read_csv('metadata-big.csv', index_col='site_id')  # this version has SMILES strings
	metadata['selfies'] = None

	print(metadata)
	newDF = []
	for rownum, row in metadata.iterrows():
		#if math.isnan(row.SMILES): row.SMILES = None
		if type(row.SMILES) != str : row.SMILES = None
		if row.SMILES:
			#print(row.SMILES)
			smilesParts = row.SMILES.split(' ')
			assert len(smilesParts)<=2
			row.selfies = sf.encoder(smilesParts[0])
		newDF.append(row)

	metadata = pandas.DataFrame(newDF)

	metadata.to_csv('metadata-selfies.csv', index_label='site_id')
	print(metadata.selfies)

	buildAlphabet(metadata)


def buildAlphabet(metadata):
	#metadata = pandas.read_csv('metadata-selfies.csv', index_col='site_id')  # this version has SMILES strings

	allSelfies = list(metadata.selfies)
	allSelfies = [l for l in allSelfies if type(l)==str]
	alphabet = sf.get_alphabet_from_selfies(allSelfies)

	vocab_stoi = {word:n+2 for n,word in enumerate(alphabet)}
	vocab_stoi['[nop]'] = 0
	vocab_stoi['.'] = 1

	with open('alphabet.txt', 'w') as f:
		json.dump(vocab_stoi, f)

	print(vocab_stoi)




def buildModel(args, alphabetLen, seqLen):
	#Embedding
	#alphabetLen, seqLen = oneHotShape

	vecShape = (seqLen, alphabetLen)


	##### Build the encoder

	inp = Input(shape=vecShape, name='onehotIn')
	x=inp
	x = Flatten()(x)
	x = Dense(32, activation='tanh')(x)
	embedding = Dense(args.fc1, activation='linear')(x)

	##### Build the decoder

	inpDec = Input(shape=(args.fc1, ), name='embeddingIn')
	x = inpDec
	x = Dropout(args.dropout)(x)
	x = Dense(32, activation='tanh')(x)
	x = Dense(seqLen*alphabetLen, activation='sigmoid')(x)

	#condition = Dense(1, activation='sigmoid', name='prediction')(x)
	outp = Reshape(vecShape)(x)
	#outp = x
	#outp = Dense(1, activation='linear', name='onehotOut')(x)

	encoder = Model(inputs=[inp], outputs=[embedding])
	decoder = Model(inputs=[inpDec], outputs=[outp])

	foo = encoder(inp)

	outp = decoder(encoder(inp))

	autoencoder = Model(inputs=[inp], outputs=[outp])


	#model = Model(inputs=[inp], outputs=[outp])
	#model = Model(inputs=[inp], outputs=[outp])
	#model.compile(loss='mse', optimizer='adam')

	#loss = 'binary_crossentropy'
	#loss = 'categorical_crossentropy'
	loss = 'mse'
	autoencoder.compile(loss=loss, optimizer='adam', metrics=['mse'])
	#model.compile(loss='binary_crossentropy', optimizer=args.opt, metrics=['acc'])
	try: plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
	except: print("NOTE: Couldn't render model.png")

	print(autoencoder.summary())

	return autoencoder, encoder, decoder




def train(args):
	metadata = pandas.read_csv('metadata-selfies.csv', index_col='site_id')  # this version has SMILES strings
	with open('alphabet.txt', 'r') as f: vocab_stoi = json.load(f)

	vocab_itos = {v:k for k,v in vocab_stoi.items()}

	uniqueDrugs = metadata[['treatment','selfies']].drop_duplicates().dropna()

	vectors = []
	maxLen = 0
	for selfie in set(uniqueDrugs.selfies):
		#if type(selfie)!=str: continue
		encoding = sf.selfies_to_encoding(selfie, vocab_stoi)
		labels, oneHot = encoding
		maxLen = max(maxLen, len(oneHot))
		print(len(oneHot))
		vectors.append(oneHot)

	alphabetLength = len(vocab_stoi)

	# pad each vector in the dataset with [nop] vectors
	dataset = []
	for vector in vectors:
		vector = np.asarray(vector)
		arr = np.zeros((maxLen, alphabetLength))
		# set each position to [nop]
		arr[:,0] = 1
		#print(arr)
		# plug the molecules oneHot roughly in the center
		vecLen = vector.shape[0]
		leftPad = (maxLen - vecLen)//2
		arr[leftPad:leftPad+vecLen, :] = vector

		dataset.append(arr)

	#dataset = np.vstack(dataset)
	dataset = np.asarray(dataset)
	datum = dataset[0]

	autoencoder, encoder, decoder = buildModel(args, alphabetLength, maxLen)

	autoencoder.fit(dataset, dataset, batch_size=1, epochs=10)


	autoencoder.save('autoencoder.h5')
	encoder.save('encoder.h5')
	decoder.save('decoder.h5')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	#parser.add_argument('-cuda', action='store_true', help='Use CUDA accelerated tSNE. (never got this working)')
	parser.add_argument('-convert', action='store_true', help='')
	parser.add_argument('-alphabet', action='store_true', help='')
	parser.add_argument('-train', action='store_true', help='')
	parser.add_argument('-fc1', type=int, default=8, help='')
	parser.add_argument('-batchSize', type=int, default=16, help='')
	parser.add_argument('-dropout', type=float, default=0.1, help='')
	args = parser.parse_args()

	if args.convert: convertSelfies()
	if args.alphabet:
		metadata = pandas.read_csv('metadata-selfies.csv', index_col='site_id')  # this version has SMILES strings
		buildAlphabet(metadata)

	if args.train: train(args)



