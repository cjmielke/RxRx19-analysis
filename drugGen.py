import random

import selfies
import numpy as np

def get_gaussian_latents(nb_latents, length, filter_latents=False):
	latents = np.random.randn(nb_latents, length).astype(np.float32)
	#if filter_latents: latents = ndimage.gaussian_filter(latents, [filter_latents, 0], mode='wrap')
	latents /= np.sqrt(np.mean(latents ** 2))
	return latents



import selfies as sf

benzene = "c1ccccc1"
gs441524 = 'C1=C2C(=NC=NN2C(=C1)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N'


# SMILES --> SELFIES translation
encoded_selfies = sf.encoder(gs441524)  # '[C][=C][C][=C][C][=C][Ring1][Branch1_2]'

# SELFIES --> SMILES translation
decoded_smiles = sf.decoder(encoded_selfies)  # 'C1=CC=CC=C1'

#len_benzene = sf.len_selfies(encoded_selfies)  # 8

#symbols_benzene = list(sf.split_selfies(encoded_selfies))

# this alphabet must be stable accross runs. Maybe its best to compute it on entire set of SMILES strings
alphabet = sf.get_alphabet_from_selfies([encoded_selfies])
vocab = {word:n for n,word in enumerate(alphabet)}

encoding = sf.selfies_to_encoding(encoded_selfies, vocab)
labels, oneHot = encoding

'''
oneHot = np.asarray(oneHot).astype(float)
noise = np.random.randn(*oneHot.shape)
oneHot+=noise
oneHot = oneHot.round().astype(int).tolist()
sf.encoding_to_selfies(oneHot, vocab, 'one_hot')
'''

def noisyOneHot(labels):
	maxLabel = max(labels)
	possible = range(0, maxLabel)
	new=[]
	mutated = random.choice(range(0,len(labels)))
	for index, elem in enumerate(labels):
		#if random.random() > 0.9: elem = random.choice(possible)
		if index==mutated : elem = random.choice(possible)
		new.append(elem)
	return new

vocab_itos = {v:k for k,v in vocab.items()}

for x in range(0, 50):
	newLabels = noisyOneHot(labels)
	selfie = sf.encoding_to_selfies(newLabels, vocab_itos, 'label')
	print(sf.decoder(selfie))
	#print(newLabels)

'''
Exiting models :
[imgEmbedding] -> mock/infected
many to one, certainly!


this one-hot encoding will be completely dependent on alphabet, so without training, these embeddings are useless
So its time to think about models

desired : generation of new drugs.
	- IE, a selfies is the output from some model, eventually.
	- Alternatively, could have a model with selfies as the input, and then optimize that input to satisfy some output 

Model 1 )
Could train a classifier like the following : 
[smiles] -> [selfies] -> [sigmoid] -> homeostatisLike

Where the output is some class that describes the drug as one that produces a homeostatis-like state.
Could also do a regression, where the output is the euclidean distance between the imaging-embedding from the homeostatic "center"

Model 2 )
Could train an autoencoder to squeeze the dimension on the selfies strings
[selfies] -> drugEmbedding -> [selfies]
drugEmbedding -> imgEmbedding

And then do tSNE on the lower-dimensional manifold. Dots could be colored. 

Model 3)
Could there be mappings between the selfies vectors and the image embeddings?
[selfies] -> [imgEmbedding]             This could be one-to-one..... maybe ?.....
[imgEmbedding] -> [selfies]             Id expect this to be a one-to-many relationship

For each of these scenarios, we could map to the embedding vectors themselves, or we could map to euclidean distances, or define sets

'''




