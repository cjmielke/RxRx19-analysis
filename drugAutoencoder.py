import argparse
import json

import pandas
from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Dropout, Reshape, \
	Activation, BatchNormalization, TimeDistributed, GRU
from keras.optimizers import SGD
from keras.utils import plot_model

import numpy as np

import selfies as sf


#sparse_categorical_crossentropy()

#categorical_crossentropy

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#from keras import metrics
from scipy.stats import stats
from sklearn.decomposition import PCA

from junk import word_acc
from plots import scatter

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






def buildLSTMModel(args, alphabetLen, seqLen):

	vecShape = (seqLen, alphabetLen)

	##### Build the encoder

	inp = Input(shape=vecShape, name='onehotIn')
	x=inp

	#dl = Dense(8, activation='tanh')
	#x = TimeDistributed(dl)(x)

	x = Flatten()(x)
	x = Dense(100, activation='relu')(x)
	x = Dense(100, activation='relu')(x)
	x = Dense(100, activation='relu')(x)
	x = Dense(args.fc1, activation='linear')(x)

	#x = LSTM(args.fc1, activation='relu', input_shape=vecShape)(x)
	#x = GRU(args.fc1, activation='linear', input_shape=vecShape)(x)
	#RepeatVector

	embedding = x

	inpDec = Input(shape=(args.fc1,), name='embeddingIn')

	x = Dense(seqLen*4, activation='relu')(inpDec)
	x = Reshape((seqLen, 4))(x)

	#x = LSTM(8, activation='relu', return_sequences=True)(x)
	x = GRU(8, activation='relu', return_sequences=True)(x)

	dl = Dense(alphabetLen, activation='softmax')

	x = TimeDistributed(dl)(x)

	outp = x

	#x = Dense(32, activation='tanh')(x)
	#x = Dense(seqLen*alphabetLen, activation='sigmoid')(x)

	#condition = Dense(1, activation='sigmoid', name='prediction')(x)
	#outp = Reshape(vecShape)(x)
	#outp = Dense(1, activation='linear', name='onehotOut')(x)

	encoder = Model(inputs=[inp], outputs=[embedding])
	decoder = Model(inputs=[inpDec], outputs=[outp])

	print(encoder.summary())
	print(decoder.summary())

	outp = decoder(encoder(inp))

	autoencoder = Model(inputs=[inp], outputs=[outp])


	#model = Model(inputs=[inp], outputs=[outp])
	#model = Model(inputs=[inp], outputs=[outp])
	#model.compile(loss='mse', optimizer='adam')

	#loss = 'binary_crossentropy'
	loss = 'categorical_crossentropy'
	#loss = 'mse'
	opt = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
	opt = 'adam'
	autoencoder.compile(loss=loss, optimizer=opt, metrics=['mse', word_acc, 'acc'])
	#model.compile(loss='binary_crossentropy', optimizer=args.opt, metrics=['acc'])
	try: plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
	except: print("NOTE: Couldn't render model.png")



	return autoencoder, encoder, decoder



def buildModel(args, alphabetLen, seqLen, convolutional=True):
	#Embedding
	#alphabetLen, seqLen = oneHotShape

	vecShape = (seqLen, alphabetLen)
	print(seqLen)

	##### Build the encoder

	inp = Input(shape=vecShape, name='onehotIn')
	x=inp

	if convolutional==True:

		x = Conv1D(16, kernel_size=3)(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(2, padding='same')(x)
		x = BatchNormalization()(x)

		x = Conv1D(32, kernel_size=3)(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(3, padding='same')(x)
		x = BatchNormalization()(x)

		x = Conv1D(32, kernel_size=3)(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(3, padding='same')(x)
		x = BatchNormalization()(x)

		'''
		x = Conv1D(32, kernel_size=3)(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(2, padding='same')(x)
		x = BatchNormalization()(x)


		x = Conv1D(32, kernel_size=3)(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(2, padding='same')(x)
		x = BatchNormalization()(x)
		'''

		x = Flatten()(x)

		#embedding = Dense(args.fc1, activation='linear')(x)
		embedding = Dense(args.fc1, activation='tanh')(x)

		##### Build the decoder

		inpDec = Input(shape=(args.fc1, ), name='embeddingIn')
		x = inpDec
		x = Dropout(args.dropout)(x)

		x = Dense(seqLen//4 * 8, activation='sigmoid')(x)
		x = Reshape((seqLen//4, 8))(x)

		x = UpSampling1D(2)(x)
		x = Conv1D(16, kernel_size=5, padding='same')(x)

		x = UpSampling1D(2)(x)
		x = Conv1D(16, kernel_size=5, padding='same')(x)


		x = Conv1D(alphabetLen, kernel_size=1)(x)
		#x = UpSampling1D(2)(x)

		x = Activation('softmax')(x)
		#x = Activation(Softmax(axis=-1))(x)


	else:

		x = Flatten()(x)

		# x = Dense(32, activation='tanh')(x)
		embedding = Dense(args.fc1, activation='linear')(x)

		##### Build the decoder

		inpDec = Input(shape=(args.fc1,), name='embeddingIn')
		x = inpDec
		x = Dropout(args.dropout)(x)

		x = Dense(seqLen * alphabetLen, activation='sigmoid')(x)
		x = Reshape(vecShape)(x)

	outp = x

	#x = Dense(32, activation='tanh')(x)
	#x = Dense(seqLen*alphabetLen, activation='sigmoid')(x)

	#condition = Dense(1, activation='sigmoid', name='prediction')(x)
	#outp = Reshape(vecShape)(x)
	#outp = Dense(1, activation='linear', name='onehotOut')(x)

	encoder = Model(inputs=[inp], outputs=[embedding])
	decoder = Model(inputs=[inpDec], outputs=[outp])


	outp = decoder(encoder(inp))

	autoencoder = Model(inputs=[inp], outputs=[outp])


	#model = Model(inputs=[inp], outputs=[outp])
	#model = Model(inputs=[inp], outputs=[outp])
	#model.compile(loss='mse', optimizer='adam')

	#loss = 'binary_crossentropy'
	loss = 'categorical_crossentropy'
	#loss = 'mse'
	opt = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
	opt = 'adam'
	autoencoder.compile(loss=loss, optimizer=opt, metrics=['mse', word_acc, 'acc'])
	#model.compile(loss='binary_crossentropy', optimizer=args.opt, metrics=['acc'])
	try: plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
	except: print("NOTE: Couldn't render model.png")

	print(encoder.summary())
	print(decoder.summary())

	return autoencoder, encoder, decoder


class DataSet():
	def __init__(self):
		self.metadataDF = None
		self.drugsDF = None
		self.alphabet = None
		self.vectorLength = None

	def encode(self):
		'''with dataframe containing selfies strings, encode them as oneHot'''


# TODO - make this object oriented
def prepData():
	metadata = pandas.read_csv('metadata-selfies.csv', index_col='site_id')  # this version has SMILES strings
	with open('alphabet.txt', 'r') as f: vocab_stoi = json.load(f)

	vocab_itos = {v:k for k,v in vocab_stoi.items()}

	uniqueDrugs = metadata[['treatment','selfies']].drop_duplicates().dropna().set_index('treatment')

	uniqueDrugs['labels'] = None
	uniqueDrugs['oneHot'] = None

	maxLen = 0
	newDF = []
	for rowNum, row in uniqueDrugs.iterrows():
		#if type(selfie)!=str: continue
		encoding = sf.selfies_to_encoding(row.selfies, vocab_stoi)
		row.labels, oneHot = encoding
		row.oneHot = np.asarray(oneHot)
		maxLen = max(maxLen, len(row.oneHot))
		#print(len(row.oneHot))
		newDF.append(row)


	uniqueDrugs = pandas.DataFrame(newDF)

	lengths = np.asarray([len(oneHot) for oneHot in uniqueDrugs.oneHot])
	assert max(lengths) == maxLen
	print(lengths.mean(), lengths.std())

	alphabetLength = len(vocab_stoi)

	seqLen = maxLen
	seqLen = 236


	def padVec(vector):
		arr = np.zeros((seqLen, alphabetLength))
		# set each position to [nop]
		arr[:,0] = 1
		#print(arr)
		# plug the molecules oneHot roughly in the center
		vecLen = vector.shape[0]
		if vecLen < seqLen:
			leftPad = (seqLen - vecLen)//2
			arr[leftPad:leftPad+vecLen, :] = vector
		else:   # truncate it
			arr[0:seqLen, :] = vector[0: seqLen, :]
		return arr


	# with the length of the maximum sequence, we can build the dataset, taking care to pad all other sequences with [nop]
	newDF = []
	for rownum, row in uniqueDrugs.iterrows():
		row.oneHot = padVec(row.oneHot)
	uniqueDrugs = pandas.DataFrame(uniqueDrugs)

	return uniqueDrugs, alphabetLength, seqLen


def train(args):

	drugsDF, alphabetLength, maxLen = prepData()

	dataset = np.asarray(list(drugsDF.oneHot))

	#autoencoder, encoder, decoder = buildModel(args, alphabetLength, maxLen)
	autoencoder, encoder, decoder = buildLSTMModel(args, alphabetLength, maxLen)

	try: autoencoder.fit(dataset, dataset, batch_size=64, epochs=999999)
	except: print('killed')
	finally:
		autoencoder.save('autoencoder.h5')
		encoder.save('encoder.h5')
		decoder.save('decoder.h5')


def embed(args):

	drugsDF, alphabetLength, maxLen = prepData()

	from keras.engine.saving import load_model
	encoder = load_model('encoder.h5')
	autoencoder = load_model('autoencoder.h5', custom_objects=dict(word_acc=word_acc))

	'''
	embeddings = []
	for rownum, row in drugsDF.iterrows():
		oneHot = np.expand_dims(row.oneHot, 0)
		embedding = encoder.predict(oneHot)
		#row['drugEmbedding'] = embedding
		embeddings.append(embedding[0])			# note, could send them in all in one batch

	embeddings = np.asarray(embeddings)
	'''

	index = drugsDF.index
	oneHots = np.asarray(list(drugsDF.oneHot))

	reproductions = autoencoder.predict(oneHots)

	labelsOrig = np.argmax(oneHots, axis=-1)

	labels = np.argmax(reproductions, axis=-1)



	#sf.encoding_to_selfies(labels.tolist(), vocab_itos=vo)

	embeddings = encoder.predict(oneHots)

	DF = pandas.DataFrame(index=index)


	if args.embed == 'latents':
		assert embeddings.shape[1]==2, 'this option only works if the autoencoder used a latent vector of length 2'
		X_embedded = embeddings
		DF['x'], DF['y'] = X_embedded[:, 0], X_embedded[:, 1]

	elif args.embed == 'tsne':
		from sklearn.manifold import TSNE
		tSNE = TSNE(n_components=2, verbose=1, n_jobs=-1, perplexity=10)
		X_embedded = tSNE.fit_transform(embeddings)
		DF['x'], DF['y'] = X_embedded[:, 0], X_embedded[:, 1]

	elif args.embed=='pca':
		X_embedded = PCA(n_components=2).fit_transform(embeddings)

		DF['x'], DF['y'] = X_embedded[:, 0], X_embedded[:, 1]
		# remove outliers
		DF = DF[(np.abs(stats.zscore(DF)) < 3).all(axis=1)]


	print('embedding maximi', DF.x.max(), DF.y.max())

	# desire a dataframe that combines information on the drugs effectiveness in each experiment
	drugPredictions = pandas.read_csv('drugPredictions.tsv', sep='\t')

	#bestDrugs = drugPredictions.groupby('treatment').max().sort_values('prediction', ascending=False)
	bestDrugs = drugPredictions.groupby('treatment').mean().sort_values('prediction', ascending=False)
	#bestDrugs = drugPredictions.groupby('treatment').median().sort_values('prediction', ascending=False)

	#joined = drugPredictions.join(DF, on='treatment', how='left')
	joined = bestDrugs.join(DF, on='treatment', how='left')


	joined = joined.dropna()

	#print(joined)

	#joined['predhit'] = (joined.prediction > 0.8)*1
	joined['predhit'] = joined.prediction.round(2)

	scatter(joined, 'predhit', title='Chemical structure embedding (%s)'%str(len(joined)), pointSize=1.0)



	'''
	drugsDF['x'], drugsDF['y'] = None, None

	for rownum, row in drugsDF.iterrows():
		embedding = X_embedded[rownum]
		row['x'] = embedding[0]
		row['y'] = embedding[1]
	'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	#parser.add_argument('-cuda', action='store_true', help='Use CUDA accelerated tSNE. (never got this working)')
	parser.add_argument('-convert', action='store_true', help='')
	parser.add_argument('-alphabet', action='store_true', help='')
	parser.add_argument('-train', action='store_true', help='')
	parser.add_argument('-embed', type=str, default=None, help='either tsne or pca')
	parser.add_argument('-fc1', type=int, default=3, help='')
	parser.add_argument('-batchSize', type=int, default=64, help='')
	parser.add_argument('-dropout', type=float, default=0.1, help='')
	args = parser.parse_args()

	if args.convert: convertSelfies()
	if args.alphabet:
		metadata = pandas.read_csv('metadata-selfies.csv', index_col='site_id')  # this version has SMILES strings
		buildAlphabet(metadata)

	if args.train: train(args)

	if args.embed:
		metadata = pandas.read_csv('metadata-selfies.csv', index_col='site_id')  # this version has SMILES strings
		embed(args)


