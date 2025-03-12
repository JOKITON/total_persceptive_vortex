import numpy as np
import scipy.linalg
from sklearn.covariance import LedoitWolf

class CSP:
	""" Extract discriminative features for binary classification tasks """

	def __init__(self, n_components=4, reg=None, log=True, norm_trace=True):
		self.n_components = n_components
		self.reg = reg
		self.log = log
		self.norm_trace = norm_trace
		self.filters_ = None  # CSP spatial filters
		self.IS_FIT = False

	def _compute_covariance(self, data):
		""" Compute covariance matrix with optional Ledoit-Wolf regularization """
		if self.reg == "ledoit_wolf":
			return LedoitWolf().fit(data).covariance_
		else:
			C = np.dot(data, data.T)  # Compute covariance
			C /= np.trace(C)  # Normalize by trace
			return (C + C.T) / 2  # Ensure symmetry

	def fit(self, epochs_data, labels):

		#** Step 1: Compute Covariance Matrices for Each Class
		unique_labels = np.unique(labels)
		class_covariances = {}

		for label in unique_labels:
			class_data = epochs_data[labels == label]
			covariances = [self._compute_covariance(trial) for trial in class_data]
			class_covariances[label] = np.mean(covariances, axis=0)

		#** Step 2: Compute Composite Covariance Matrix
		C1 = class_covariances[unique_labels[0]]
		C2 = class_covariances[unique_labels[1]]
		# print("C1, C2:", C1.shape, C2.shape)

		C_comp = C1 + C2
		# print("C composite:", C_comp.shape)

		#** Step 3: Solve Generalized Eigenvalue Problem
		eigenvalues, eigenvectors = scipy.linalg.eigh(C1, C_comp)
		#? (eigenvalues).shape : (433,)
		#? (eigenvectors).shape : (433, 433)
		# print(eigenvalues.shape, eigenvectors.shape)
		# print("Eigenvalues:", eigenvalues)

		#* Sort eigenvalues in descending order
		sorted_indices = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[sorted_indices]
		eigenvectors = eigenvectors[:, sorted_indices]

		# Select the top and bottom n_components
		self.filters_ = np.hstack((eigenvectors[:, :self.n_components],
								eigenvectors[:, -self.n_components:]))
		#? (filters_).shape : (n_times, 2*n_components)

		self.IS_FIT = True

	def transform(self, data):
		if self.IS_FIT is False:
			raise(ValueError("Error: Data has not been proccessed before. Use .fit() method before .tranform()..."))
		#** Step 4: Apply the CSP filters to the data

		features = np.array([np.dot(self.filters_.T, trial.T) for trial in data])
		#? (trial).shape : (n_channels, n_times)
		#? (features) -> (2*n_components, n_channels)
		# print("CSP Filters Shape:", self.filters_.shape)

		#* Compute log variance
		# print("Features before:", features.shape)
		#? (features) -> (n_epochs, 2*n_components, n_channels)
		features = np.log(np.var(features, axis=2))
		# print("Features after:", features.shape)
		#? (features) -> (n_epochs, 2*n_components)
		return features

	def fit_transform(self, epochs_data, labels):
		""" Apply CSP to the filtered data """

		self.fit(epochs_data, labels)
		features = self.transform(epochs_data)
		return features
