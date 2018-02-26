import os
import os.path
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

class PCA():

	def __init__(self, path_data):
		
		try:
			self._df = loadmat(path_data)
		except IOError:
			print("No su file exist")
			raise

		self.X = self._df['X']

	def getX(self):

		return (self.X)

	def plotData(self, X,  mu=1, s=1, u=1, type=0):

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
		ax.set_xlim(0.5, 6.5)
		ax.set_ylim(2, 8)
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
		# U = np.seros(n)
		# S = np.zeros(n)

		covariance = np.dot(X.T, X) / m
		U, S, VH = np.linalg.svd(covariance)

		return (U, S, VH)



def main():
	
	"""
	# ====================== Part 1: Load Examples Data =======================
	"""

	current_path = os.getcwd()
	path_data = current_path + '/ex7data1.mat'
	pca = PCA(path_data)

	# pca.plotData(pca.getX())

	"""
	# ==================== Part 2: Prinipal Components Analysis ===============
	"""

	X_norm, mu, sigma = pca.featureNormalize(pca.getX())
	u, s, _ = pca.pca(X_norm)
	print(s)
	print(u)
	pca.plotData(pca.getX(), mu, s, u, 1)



if __name__ == "__main__":
	main()