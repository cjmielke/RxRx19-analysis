import argparse

import pandas

from norm import normalizeColumns

parser = argparse.ArgumentParser(description='compute tSNE embeddings')
parser.add_argument('-cuda', action='store_true', help='Use CUDA accelerated tSNE. (never got this working)')
parser.add_argument('-fraction', type=float, default=0.05, help='The fraction of the dataset to perform tSNE clustering on. Defaults to 5%. The full dataset takes many hours to cluster.')
args = parser.parse_args()


metadata = pandas.read_csv('metadata.csv', index_col='site_id')

print('Loading hdf5')
embeddings = pandas.read_hdf('embeddings.hdf','df')
print('Loaded')

# Try normalizing the embeddings data between experiments

normalize = ['experiment', 'site', 'plate', 'disease_condition']
#normalize = False

if normalize: embeddings = normalizeColumns(metadata, embeddings, columns=normalize)



# try focusing just on experiment 1, effectively cutting the experiment in half
#exp1 = metadata[metadata.experiment=='HRCE-1']
#embeddings = embeddings[embeddings.index.isin(set(exp1.index))]



# shuffle the dataset, and select a fraction of it for tSNE clustering.
embeddings = embeddings.sample(frac=args.fraction)


X = embeddings.values
index = embeddings.index

print('Fitting and transforming tSNE')

if args.cuda:
	from tsnecuda import TSNE  # FASTER!!!
	tSNE = TSNE(n_components=2, verbose=1, perplexity=50)
else:
	from sklearn.manifold import TSNE
	tSNE = TSNE(n_components=2, verbose=1, n_jobs=-1, perplexity=5)

X_embedded = tSNE.fit_transform(X)
#tSNE.fit_transform()


DF = pandas.DataFrame(index=index)

DF['x'] = X_embedded[:,0]
DF['y'] = X_embedded[:,1]

DF = DF.join(metadata)

# some of the embeddings couldn't be joined to an associated entry in the metadata file! Had to remove those ...

DFnan = DF[DF.disease_condition.isna()]
DF = DF[~DF.disease_condition.isna()]

print(len(DF), 'joined')
print(len(DFnan), 'nans')


DF.to_hdf('tSNE.hdf', 'df')


