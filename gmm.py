# borrowed from https://github.com/vlavorini/ClusterCardinality/blob/master/Cluster%20Cardinality.ipynb


import numpy as np
import matplotlib.pyplot as plt
import pandas
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture as GMM
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import rcParams

rcParams['figure.figsize'] = 16, 8


def draw_ellipse(position, covariance, ax=None, **kwargs):
	"""Draw an ellipse with a given position and covariance"""
	ax = ax or plt.gca()
	# Convert covariance to principal axes
	if covariance.shape == (2, 2):
		U, s, Vt = np.linalg.svd(covariance)
		angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
		width, height = 2 * np.sqrt(s)
	else:
		angle = 0
		width, height = 2 * np.sqrt(covariance)

	# Draw the Ellipse
	for nsig in range(1, 4):
		ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
	ax = ax or plt.gca()
	labels = gmm.fit(X).predict(X)
	if label:
		ax.scatter(X[:, 0], X[:, 1], c=labels, s=0.1, cmap='viridis', zorder=2)
	else:
		ax.scatter(X[:, 0], X[:, 1], s=1, zorder=2)

	w_factor = 0.2 / gmm.weights_.max()
	for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
		draw_ellipse(pos, covar, alpha=w * w_factor)
	plt.title("GMM with %d components" % len(gmm.means_), fontsize=(20))
	plt.xlabel("U.A.")
	plt.ylabel("U.A.")

	plt.savefig('gmm.png')


def SelBest(arr: list, X: int) -> list:
	'''
	returns the set of X configurations with shorter distance
	'''
	dx = np.argsort(arr)[:X]
	return arr[dx]




df = pandas.read_hdf('tSNE.hdf','df')
normalCells = df[df.disease_condition != 'Active SARS-CoV-2']
normalPoints = normalCells[['x','y']].values

embeddings = normalPoints



gmm=GMM(15).fit(embeddings)
plot_gmm(gmm, embeddings)



