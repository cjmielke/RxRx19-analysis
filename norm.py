import pandas



def normalizeColumns(metadata, embeddings, columns=None):

	columns = columns or ['experiment', 'site', 'plate', 'disease_condition']

	for colName in columns:

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

	return embeddings