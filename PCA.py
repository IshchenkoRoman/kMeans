import os
import os.path
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

class PCA():

	def __init__(self, path_data, path_faces):
		
		try:
			self._df = loadmat(path_data)
		except IOError:
			print("No such file exist")
			raise

		try:
			self._dfFaces = loadmat(path_faces)
		except IOError:
			print("No data faces exist")
			raise


		self.X = self._df['X']
		self.Xf = self._dfFaces['X']

	def getX(self):

		return (self.X)

	def getXFaces(self):

		return (self.Xf)

	def plotData(self, X, x_lim=(None, None), y_lim=(None, None), mu=1, s=1, u=1, type=0):

		fig, ax = plt.subplots()

		data = ax.scatter(X[:,0], X[:,1], marker='o', c='w', edgecolor='b', label="Training data")

		if type==1:

			l1 = mu + 1.5 * s[0] * u[:,0]
			l2 = mu + 1.5 * s[1] * u[:,1]

			ax.plot([mu[0], l1[0]], [mu[1], l1[1]], '-k', linewidth=2)
			ax.plot([mu[0], l2[0]], [mu[1], l2[1]], '-k', linewidth=2)

		handlers, labels = ax.get_legend_handles_labels()
		hl = sorted(zip(handlers, labels), key=operator.itemgetter(1))
		h,l = zip(*hl)
		ax.legend(h, l, loc=2, framealpha=1)
		ax.set_title("Visualise data")
		ax.set_xlim(x_lim[0], x_lim[1])
		ax.set_ylim(y_lim[0], y_lim[1])
		plt.show()

	def featureNormalize(self, X):

		mu = np.mean(X, axis=0)
		X_norm = X - mu

		# STD NUMPY != STD MATPLOTLIB
		# https://stackoverflow.com/questions/27600207/why-does-numpy-std-give-a-different-result-to-matlab-std/27600240#27600240
		sigma = np.std(X_norm, axis=0, ddof=1)
		X_norm = X_norm / sigma

		return (X_norm, mu, sigma)

	def pca(self, X):

		m, n = X.shape

		covariance = np.dot(X.T, X) / m
		U, S, VH = np.linalg.svd(covariance)

		return (U, S, VH)

	def projectData(self, X, U, K):

		"""
		Compute the projection of the normilized inputs X into the reduced
		dimensional space spanned by the first K columns of U. It return the 
		project examples in Z.
		It compute the projection of the data using only the top K eigenvectors
		in U (first K columns).
		"""

		# print(X)
		# print(U)
		# print(X.shape, U.shape)
		U_reduce = U[:,0:K]
		Z = np.dot(X, U_reduce)
		return (Z)



	def recoverData(self, Z, U, K):

		"""
		Recovers an aproximation the original data, that has benn reduced
		to K dimensions. It return approximate reconstruction in X_rec.
		"""

		X_rec = np.zeros((Z.shape[0], U.shape[0]))
		print(X_rec.shape)

		for i in range(X_rec.shape[0]):

			v = Z[i,:]
			for j in range(U.shape[0]):
				X_rec[i][j] = np.dot(v, U[j, :K])

		return (X_rec)

	def plotProjection(self, X_norm, X_rec):

		fig, ax = plt.subplots()

		pdl = ax.scatter(X_rec[:,0], X_rec[:,1], marker='o', c='w', edgecolor='r', label="Projected data")
		ndl = ax.scatter(X_norm[:,0], X_norm[:,1], marker='o', c='w', edgecolor='b', label="Normalized data")

		# Plot projection
		for i in range(X_norm.shape[0]):
			ax.plot([X_norm[i][0], X_rec[i][0]], [X_norm[i][1], X_rec[i][1]], '--k', linewidth=1)

		pll = ax.plot([X_norm[0][0], X_rec[0][0]], [X_norm[0][1], X_rec[0][1]], '--k', linewidth=1, label="Projected line")

		handler, labels = ax.get_legend_handles_labels()
		hl = sorted(zip(handler, labels), key=operator.itemgetter(1))
		h, l = zip(*hl)
		ax.legend(h, l, loc=2, framealpha=0.5)

		ax.set_title("The normalized and projected data after PCA.")
		ax.set_xlim(-4, 3)
		ax.set_ylim(-4, 3)
		plt.show()

	def plotFaces(self, X):

		fig = plt.figure()

		for i in range(1, 11):
			fig.add_subplot(10, 1, i)
			array = X[(i - 1) * 10:(i) * 10].reshape(-1, 32).T
			plt.imshow(array, cmap="gray")
			plt.axis("off")

		plt.suptitle("First 100 faces")
		plt.show()

		# for i in range(10):
		# 	for j in range(10):
		# 		im = X[i].reshape(32, 32)
		# 		ax[i, j].imshow(im)

		plt.show()




def main():
	
	"""
	# ====================== Part 1: Load Examples Data =======================
	"""

	current_path = os.getcwd()
	path_data = current_path + "/ex7data1.mat"
	path_faces = current_path + "/ex7faces.mat"
	pca = PCA(path_data, path_faces)

	# pca.plotData(pca.getX())

	"""
	# ==================== Part 2: Prinipal Components Analysis ===============
	"""

	X_norm, mu, sigma = pca.featureNormalize(pca.getX())
	u, s, _ = pca.pca(X_norm)
	print(s)
	print(u)
	# pca.plotData(pca.getX(), (0.5, 6.5), (2, 8), mu, s, u, 1)
	# pca.plotData(pca.getX(), mu, s, u, 1)

	"""
	# =================== Part 3: Dimension Reduction =========================
	# Implement projection step to map the data onto the first k eigenvectors.
	# Then plot the data in tthis reduced dimension spac. This will show what data
	# looks like, when using inly correspoding eigenvectors to reconstruct it.
	"""

	# pca.plotData(X_norm, (-4, 3), (-4, 3))
	K = 1
	Z = pca.projectData(X_norm ,u, K)
	X_rec = pca.recoverData(Z, u, K)
	# pca.plotProjection(X_norm, X_rec)

	"""
	# =================== Part 4: Loading and Visualizing FAce Data ===========
	"""

	print(pca.Xf.shape)
	pca.plotFaces(pca.getXFaces())






if __name__ == "__main__":
	main()