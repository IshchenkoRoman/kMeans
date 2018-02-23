import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

class KMeans():

	def __init__(self, path_data2):
		
		try:
			self._df2 = loadmat(path_data2)
		except IOError:
			print("No such file exist")
			raise

		self.X2 = self._df2['X']

	def getX2(self):

		return (self.X2)

	def findCosestCentroids(self, X, initial_cenroids):

		index = []
		len_centorids = initial_cenroids.shape[0]

		for i in X:
			mindist = np.sum(np.power(i - initial_cenroids[0], 2))
			ind = 0
			for j in range(len_centorids):
				mind = np.sum(np.power(i - initial_cenroids[j], 2))
				if mind <= mindist:
					mindist = mind
					ind = j

			index.append(ind)

		return (index)

	def computeCentroids(self, X, idx, K):

		len_ = len(idx)
		centers = np.zeros((K, 2))
		count_idx = np.zeros((K, 1))

		for i in range(len_):

			centers[idx[i]] += X[i]
			count_idx[idx[i]] += 1

		res = np.divide(centers, count_idx)
		return (res[0])

def main():
	
	"""
	# ===================== Part 1: Find Closest Centroids ====================
	"""

	current_path = os.getcwd()

	path_data2 = current_path + "/ex7data2.mat"

	kmeans = KMeans(path_data2)
	
	# 3 Centroids
	K = 3
	initial_cenroids = np.array([[3, 3], [6, 2], [8, 5]])
	idx = kmeans.findCosestCentroids(kmeans.getX2(), initial_cenroids)
	print(idx[:3])

	"""
	# ===================== Part 2: Compute Means =============================
	"""
	means = kmeans.computeCentroids(kmeans.getX2(), idx, K)
	print(means)


if __name__ == "__main__":
	main()