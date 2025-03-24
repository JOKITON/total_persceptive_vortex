import numpy as np
import scipy.linalg
from sklearn.covariance import LedoitWolf
from mne.decoding import CSP as orCSP

from utils.csp import apply_bandpass_filter

from sklearn.base import BaseEstimator, TransformerMixin

class CSP(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=8, freq_bands=None, fs=160, reg=None, log=True, norm_trace=False):
		self.n_components = n_components
		self.reg = reg
		self.log = log
		self.norm_trace = norm_trace
		self.freq_bands = freq_bands
		self.fs = fs
		self.csp = []
		self.IS_FIT = False

	def fit(self, epochs_data, labels):
		self.epochs_data = epochs_data
		self.labels = labels
		self.csp = []

		if self.freq_bands is None:
			csp = orCSP(n_components=self.n_components, reg=self.reg, log=self.log, norm_trace=self.norm_trace)
			csp.fit(epochs_data, labels)
			self.csp.append(csp)
			return self

		for lowcut, highcut in self.freq_bands:
			filtered_data = apply_bandpass_filter(epochs_data, lowcut, highcut, self.fs)
			csp = orCSP(n_components=self.n_components, reg=self.reg, log=self.log, norm_trace=self.norm_trace)
			csp.fit(filtered_data, labels)
			self.csp.append(csp)

		self.IS_FIT = True
		return self  # Required for `Pipeline`

	def transform(self, data):
		if not self.IS_FIT:
			raise ValueError("Error: CSP must be fitted before calling transform().")

		if self.freq_bands is None:
			return self.csp[0].transform(data)

		all_features = [
			csp.transform(apply_bandpass_filter(data, lowcut, highcut, self.fs))
			for (lowcut, highcut), csp in zip(self.freq_bands, self.csp)
		]

		return np.concatenate(all_features, axis=1)

	def fit_transform(self, epochs_data, labels):
		return self.fit(epochs_data, labels).transform(epochs_data)
