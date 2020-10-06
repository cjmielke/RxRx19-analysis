import argparse

import pandas

parser = argparse.ArgumentParser(description='compute tSNE embeddings')
parser.add_argument('-cuda', action='store_true', help='Use CUDA accelerated tSNE')
args = parser.parse_args()


metadata = '../data/RxRx19/RxRx19a/metadata-big.csv'
metadata = pandas.read_csv(metadata, index_col='site_id')

print('Loading hdf5')
embeddings = pandas.read_hdf('embeddings.hdf','df')
#embeddings = pandas.read_parquet('embeddings.pq.gz')
print('Loaded')

print(embeddings)


# Try normalizing the embeddings data between experiments

normalize = ['experiment', 'site', 'plate', 'disease_condition']
#normalize = False

if normalize:

	for colName in normalize:

		print('Normalizing by ', colName)

		parts = []
		for colVal in set(metadata[colName]):
			partition = metadata[metadata[colName]==colVal]
			embeddingsPartition = embeddings[ embeddings.index.isin(set(partition.index)) ]

			# compute centroid of embeddings vectors, and then subtract each
			centroid = embeddingsPartition.mean(axis=0)
			embeddingsPartition -= centroid
			parts.append(embeddingsPartition)

		embeddings = pandas.concat(parts)
		# now the embeddings dataframe should be normalized to each experiment



# Todo - turn this routine into a function that can be applied to any column


# try focusing just on experiment 1, effectively cutting the experiment in half
#exp1 = metadata[metadata.experiment=='HRCE-1']
#embeddings = embeddings[embeddings.index.isin(set(exp1.index))]



# process a small fraction ...
#embeddings = embeddings.sample(frac=0.05)


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

DFnan = DF[DF.disease_condition.isna()]
DF = DF[~DF.disease_condition.isna()]

print(len(DF), 'joined')
print(len(DFnan), 'nans')


DF.to_hdf('DF.hdf', 'df')



'''
colors = {'Active SARS-CoV-2':'red', 'UV Inactivated SARS-CoV-2':'blue', 'Mock':'purple'}
ax = DF.plot.scatter('x','y', c=DF['disease_condition'].apply(lambda x: colors.get(x, 'green')), s=0.2)

fig = ax.get_figure()
fig.savefig('tsne-condition.png', dpi=300)
'''


