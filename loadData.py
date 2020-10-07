import argparse


parser = argparse.ArgumentParser(description='convert and summarize the dataset')
parser.add_argument('-convert', type=str, help='Path to embeddings.csv file. Script will parse it and convert to hdf5 for faster loading later')
parser.add_argument('-summarize', action='store_true', help='print summary statistics of metadata')



import numpy as np

import pandas


# 305520 rows
metadata = pandas.read_csv('metadata.csv', index_col='site_id')

print(metadata.treatment.value_counts())
print(metadata.disease_condition.value_counts())


def convert(embeddingsFile):
	# force loading as float32 for memory efficiency
	df_test = pandas.read_csv(embeddingsFile, nrows=100)
	float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
	float32_cols = {c: np.float32 for c in float_cols}
	print('loading embeddings .....')
	embeddings = pandas.read_csv(embeddingsFile, engine='c', dtype=float32_cols, index_col='site_id')
	print('DONE loading embeddings .....')

	print(embeddings)

	embeddings.to_hdf('embeddings.hdf', 'df', mode='w')
	#embeddings.to_parquet('embeddings.pq')
	#embeddings.to_parquet('embeddings.pq.gz', compression='gzip')

def summarize():
	#embeddings = pandas.read_hdf('embeddings.hdf', 'df')

	for colname in metadata.columns:
		col = metadata[colname]
		numuniq = len(set(col.dropna()))
		print('='*10, colname, '=== unique values : ', numuniq)
		if numuniq < 50:
			print(col.value_counts())


if __name__ == '__main__':
	args = parser.parse_args()

	if args.convert: convert(args.convert)

	if args.summarize: summarize()



