import os
import os.path
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

from scipy.io import loadmat
from scipy.optimize import minimize

import kMeans as kM

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

	def plotFaces(self, X, count=10, title="First N faces"):

		fig = plt.figure()
		m, n = X.shape

		# Alternative

		# gs = gridspec.GridSpec(count, count)
		# gs.update(bottom=0.01, top=0.99, left=0.01, right=0.99, hspace=0.05, wspace=0.05)
		# k = 0
		# for i in range(count):
		# 	for j in range(count):
		# 		ax = plt.subplot(gs[i, j])
		# 		ax.axis('off')
		# 		ax.imshow(X[i + j].reshape(int(np.sqrt(n)), int(np.sqrt(n))).T, cmap=plt.get_cmap("Greys"), interpolation = "nearest")

		for i in range(1, count + 1):
			fig.add_subplot(count, 1, i)
			array = X[(i - 1) * count:(i) * count].reshape(-1, int(np.sqrt(n))).T
			plt.imshow(array, cmap="gray", interpolation="nearest")
			plt.axis("off")

		plt.suptitle(title)
		plt.show()

	def test(self, X_norm, X_rec):

		self.plotFaces(X_norm[:100])
		self.plotFaces(X_rec[:100])

	def plot3D(self, X, sel, colors):

		fig = plt.figure(1)
		ax = fig.add_subplot(111, projection="3d")

		ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=25, c=colors)
		plt.title("Pixel dataset plotted in 3D.Colors shows centroids memberships.")
		plt.show()

	def plot2D(self, X, idx, K):

		palette = colors.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, K+1), np.ones(((K+1), 2) ) ]) )
		color = np.array([palette[int(i)] for i in idx])

		plt.scatter(X[:,0], X[:,1], s=15, facecolors="none", edgecolor=color)
		plt.title("Pixels dataset plotted in 2D, using PCA for dimensionality reduction")
		plt.show()


def main():
	
	"""
	# ====================== Part 1: Load Examples Data =======================
	"""

	current_path = os.getcwd()
	path_data = current_path + "/ex7data1.mat"
	path_faces = current_path + "/ex7faces.mat"
	pca = PCA(path_data, path_faces)

	# # pca.plotData(pca.getX())

	# """
	# # ==================== Part 2: Prinipal Components Analysis ===============
	# """

	# X_norm, mu, sigma = pca.featureNormalize(pca.getX())
	# u, s, _ = pca.pca(X_norm)
	# # pca.plotData(pca.getX(), (0.5, 6.5), (2, 8), mu, s, u, 1)
	# # pca.plotData(pca.getX(), mu, s, u, 1)

	# """
	# # =================== Part 3: Dimension Reduction =========================
	# # Implement projection step to map the data onto the first k eigenvectors.
	# # Then plot the data in tthis reduced dimension spac. This will show what data
	# # looks like, when using inly correspoding eigenvectors to reconstruct it.
	# """

	# # pca.plotData(X_norm, (-4, 3), (-4, 3))
	# K = 1
	# Z = pca.projectData(X_norm ,u, K)
	# X_rec = pca.recoverData(Z, u, K)
	# # pca.plotProjection(X_norm, X_rec)

	# """
	# # =================== Part 4: Loading and Visualizing FAce Data ===========
	# """

	# # print(pca.Xf.shape)
	# # pca.plotFaces(pca.getXFaces())

	# """
	# # =================== Part 5: PCA on Face Data: Eigenfaces ================
	# # Run PCA and visualise the eigenvectirs which are in thhis case eigenfaces
	# # We display first 36 eigenfaces
	# """

	# X_norm, mu, sigma = pca.featureNormalize(pca.getXFaces())

	# U, S, VH = pca.pca(X_norm)
	# print(U.shape)
	# # pca.plotFaces(-U.T, 6, "Principal components on the face dataset")

	# """
	# # =================== Part 6: Dimension Reduction for Faces ===============
	# # Project images to the eigen space using the top k eigenvectors
	# """

	# K = 100
	# Z = pca.projectData(X_norm, U, K)

	# """
	# # =================== Part 7: Visualization of Faces after PCA Dimension Reduction 
	# # Project images to the eigen space using the top k eigenvectors 
	# """

	# X_rec = pca.recoverData(Z, U, K)
	# pca.test(X_norm, X_rec)

	"""
	# =================== Part 8(a): PCA For Visualisation =======================
	# One useful apllication of PCA is to use it to visualize high-dimensional
	# data. We'll run K-Means on 3-dimensional pixels colors of an image. 
	# Firstly visualize this output in 3D, and the apply PCA to obtain a 
	# visualization in 2D.
	"""

	path_data2 = current_path + "/ex7data2.mat"
	path_image = current_path + "/bird_small.png"
	path_image_mat = current_path + "/bird_small.mat"

	kmeans = kM.KMeans(path_data2, path_image)
	X = kmeans.getX()

	K = 16
	max_iters = 10
	initial_centroids = kmeans.KmeansInitCentroids(X, K)
	centroids, idx = kmeans.runkMeans(X, initial_centroids, max_iters)

	sel = np.floor(np.random.rand(1000, 1) * X.shape[0]).astype(int).flatten()

	palette = colors.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, K+1), np.ones(((K+1), 2) ) ]) )
	color = palette[idx[sel]]

	pca.plot3D(X, sel, color)

	"""
	# =================== Part 8(b): PCA for Visualization ====================
	"""

	X_norm, mu, sigma = pca.featureNormalize(X)

	U, S, VH = pca.pca(X_norm)

	Z = pca.projectData(X_norm, U, 2)

	pca.plot2D(Z[sel, :], idx[sel], K)


if __name__ == "__main__":
	main()