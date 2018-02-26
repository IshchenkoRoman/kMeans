import os
import os.path
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from scipy.io import loadmat
from scipy.optimize import minimize

from scipy import ndimage
from scipy import misc

import cv2
from cv2 import normalize

class KMeans():

	def __init__(self, path_data2, path_image):
		
		try:
			self._df2 = loadmat(path_data2)
		except IOError:
			print("No such file exist")
			raise

		try:
			self.A = cv2.imread(path_image)

		except IOError:
			print("No picture :(")
			raise

		self.X2 = self._df2['X']

		self.A = self.A / 255
		self.X = self.A.reshape(self.A.shape[0] * self.A.shape[1], 3, order='F').copy()

	def getX2(self):

		return (self.X2)

	def getX(self):

		return (self.X)

	def findClosestCentroids(self, X, initial_centroids):

		# https://stackoverflow.com/questions/17393989/arrays-used-as-indices-must-be-of-integer-or-boolean-type/24261734#24261734
		index = np.zeros((1, X.shape[0]), dtype=np.int8)
		len_centorids = initial_centroids.shape[0]

		k = 0
		for i in X:
			mindist = np.sum(np.power(i - initial_centroids[0], 2))
			ind = 0
			for j in range(len_centorids):
				mind = np.sum(np.power(i - initial_centroids[j], 2))
				if mind <= mindist:
					mindist = mind
					ind = j

			index[0][k] = ind
			k += 1

		return (index[0])

	def computeCentroids(self, X, idx, K):

		len_ = len(idx)
		centers = np.zeros((K, X.shape[1]))
		count_idx = np.zeros((K, 1))

		# print(X.shape, idx.shape, centers.shape)

		for i in range(len_):

			centers[int(idx[i])] += X[i]
			count_idx[int(idx[i])] += 1

		res = np.divide(centers, count_idx)
		return (res)

	def runkMeans(self, X, initial_centroids, max_iters, plot_progress=False):

		if plot_progress:
			fig, ax = plt.subplots()

		m, n = X.shape
		K = initial_centroids.shape[0]
		centroids = initial_centroids
		previous_centroids = centroids
		idx = np.zeros((m, 1))

		# if plotting, set up the space for interactive graphs
		# http://stackoverflow.com/a/4098938/583834
		# http://matplotlib.org/faq/usage_faq.html#what-is-interactive-mode

		if plot_progress:
			plt.close()
			plt.ion()

		# Run K-Means
		for i in range(max_iters):

			print("K-Means iteration {:d} of {:d}...\n".format(i, max_iters))

			idx = self.findClosestCentroids(X, centroids)

			if plot_progress:

				palette = colors.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, K+1), np.ones(((K+1), 2) ) ]) )
				color = np.array([palette[int(i)] for i in idx])

				ax.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors=color)
				l = [str(i + 1) + " cluster" for i in range(centroids.shape[0])]
				cent = ax.scatter(centroids[:,0], centroids[:,1], marker='h', s=100, c=palette, linewidth=2, edgecolor='k', label=l)

				for j in range(centroids.shape[0]):
					ax.plot([centroids[j][0], previous_centroids[j][0]], [centroids[j][1], previous_centroids[j][1]], c='k')

				ax.set_title("K-Means in action with {:d} centers, iteration {:d}".format(int(centroids.shape[0]), i + 1))

				test = ax.scatter([],[], color='y', marker='h', label="Centroids", edgecolor='k')

				leg = ax.legend(handles=[test])
				leg.legendHandles[0]._sizes = [100]
				fig.show()
				if (i < 5):
					input("Press Enter")
				else:
					input("Keep going, you can do it!")

				previous_centroids = centroids

			centroids = self.computeCentroids(X, idx, K)

		return (centroids, idx)

	def KmeansInitCentroids(self, X, K):

		centroids = np.zeros((K, X.shape[1]))

		# print(max(np.amax(X, axis=0)))
		randidx = np.random.permutation(X.shape[0])
		
		# for i in range(K):
		# 	centroids[i] =  randidx[i]

		centroids = X[randidx[:K], :]

		return (centroids)

	def plotPictures(self, origianl_picture, compressed, centroids, idx, K):

		fig, ax = plt.subplots(1, 3)
		fig.subplots_adjust(top=0.92, left=0.07, right=0.97, hspace=0.3, wspace=0.3)
		plt.subplot(1, 3, 1)
		plt.imshow(origianl_picture)
		plt.title("Original")

		plt.subplot(1, 3, 2)
		plt.imshow(compressed)
		plt.title("Compressed, with {:d} colors.".format(K))

		plt.subplot(1, 3, 3)
		plt.imshow(compressed)
		palette = colors.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, K+1), np.ones(((K+1), 2) ) ]) )
		color = np.array([palette[int(i)] for i in idx])
		cent = plt.scatter(centroids[:,0] * 128, centroids[:,1] * 128, marker='h', s=100, c=palette, linewidth=2, edgecolor='k')
		plt.title("With Centroids")
		fig.show()
		plt.show()


def main():
	
	"""
	# ===================== Part 1: Find Closest Centroids ====================
	"""
	# np.random.seed(42)
	current_path = os.getcwd()

	path_data2 = current_path + "/ex7data2.mat"
	path_image = current_path + "/bird_small.png"
	path_image_mat = current_path + "/bird_small.mat"

	kmeans = KMeans(path_data2, path_image)
	
	# 3 Centroids
	K = 3
	initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
	idx = kmeans.findClosestCentroids(kmeans.getX2(), initial_centroids)
	print(idx[:3])

	"""
	# ===================== Part 2: Compute Means =============================
	"""
	means = kmeans.computeCentroids(kmeans.getX2(), idx, K)
	# print(means)


	"""
	# ===================== Part 3: K-Means clustering ========================
	# Run K-Means algorith on example dataset that was provided
	"""	

	c, i = kmeans.runkMeans(kmeans.X2, initial_centroids, 10, plot_progress=True)
	kmeans.KmeansInitCentroids(kmeans.X2, K)

	new_centroid = kmeans.KmeansInitCentroids(kmeans.getX2(), K)
	# print(new_centroid)

	c, i = kmeans.runkMeans(kmeans.getX2(), new_centroid, 10, plot_progress=True)
	kmeans.KmeansInitCentroids(kmeans.getX2(), K)

	"""
	# ===================== Part 4: K-Means Clustering on Pixels ==============
	# Run K-Means algorith on example dataset that was based on pixels of image
	"""

	K = 16
	max_iters = 10

	initial_centroids = kmeans.KmeansInitCentroids(kmeans.getX(), K)
	centroids, idx = kmeans.runkMeans(kmeans.getX(), initial_centroids, max_iters, plot_progress=True)

	"""
	# ===================== Part 5: Image Compression =========================
	# Use K-Means for compressing image
	"""

	"""
	# Represent the image X as in terms of indices in idx
	"""

	idx = kmeans.findClosestCentroids(kmeans.getX(), centroids)

	"""
	# Now, we can recover the image from the indices (idx) by mapping each pixel 
	# (specified by it's index in idx) to the centroid value
	"""

	print(centroids)
	X_recovered = centroids[idx[:], :]
	X_recovered = X_recovered.reshape(kmeans.A.shape[0], kmeans.A.shape[1], 3, order='F')

	kmeans.plotPictures(kmeans.A, X_recovered, centroids, idx, K)



if __name__ == "__main__":
	main()