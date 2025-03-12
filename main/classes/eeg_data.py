import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mne
from mne import create_info
from mne.io import RawArray
from mne.datasets import eegbci
from mne.io.edf import read_raw_edf
from mne.decoding import UnsupervisedSpatialFilter, CSP
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from scipy.signal import butter, filtfilt
import pickle

import lightgbm as lgb

from utils.csp import butter_bandpass, apply_bandpass_filter, compute_csp
from utils.data import read_data, save_data, fetch_data, save_model, load_model, save_test_data, load_test_data
from utils.ica import remove_eog, remove_ecg

class EEGData:

	def __init__(self, config, config_csp, folder, verbose=False):
		"""Initialize EEGData object with file paths and optional filter settings."""
		self.IS_FILTERED = False
		self.IS_NORMALIZED = False
		self.IS_ICA = False

		self.raw_h, self.raw_hf = None, None
		self.raw_filt_h, self.raw_filt_hf = None, None
		self.norm_raw_h, self.norm_raw_hf = None, None
		self.clean_raw_h, self.clean_raw_hf = None, None
		self.ica_h = None
		self.ica_hf = None
		self.epochs = None
		self.lda, self.svm, self.rf = None, None, None

		# Initialize basic variables to fill later
		self.folder_path = folder
		self.config = config
		self.csp_config = config_csp

		self.subject = np.arange(config["n_subjects"]) + 1

		# Create a montage object to store the channel positions
		self.montage = mne.channels.make_standard_montage(config["montage"])

		self.load_data(folder, verbose=verbose)

		# Ensure raw_h is initialized before accessing its attributes
		if self.raw_h is not None:
			# Get the picks for the EEG channels
			self.picks = mne.pick_types(self.raw_h.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
		else:
			raise ValueError("raw_h is not initialized. Please check the data loading process.")

		# Store the frequencies for later on filtering/plotting
		self.l_freq = config["l_freq"]
		self.h_freq = config["h_freq"]

	def load_data(self, folder, verbose=False):
		"""Loads EEG data from files."""
		fast_start = self.config['fast_start'].lower() == 'true'
		is_raw_local = self.config['is_raw_local'].lower() == 'true'
		is_raw_filt_local = self.config['is_raw_filt_local'].lower() == 'true'
		is_event_local = self.config['is_event_local'].lower() == 'true'
		is_ica_local = self.config['is_ica_local'].lower() == 'true'
		is_epoch_local = self.config['is_epoch_local'].lower() == 'true'

		if is_raw_local is False: # In case data is not stored locally
			data1_h, _ = fetch_data(self.subject, self.config["run_exec_h"],
				{1:'rest', 2: 'do/left_hand', 3: 'do/right_hand'}, self.montage, verbose=verbose)
			data2_h, _ = fetch_data(self.subject, self.config["run_img_h"],
				{1:'rest', 2: 'imagine/left_hand', 3: 'imagine/right_hand'}, self.montage, verbose=verbose)

			data1_hf, _ = fetch_data(self.subject, self.config["run_exec_hf"],
				{1:'rest', 2: 'do/feet', 3: 'do/hands'}, self.montage, verbose=verbose)
			data2_hf, _ = fetch_data(self.subject, self.config["run_img_hf"],
				{1:'rest', 2: 'imagine/feet', 3: 'imagine/hands'}, self.montage, verbose=verbose)

			self.raw_h = mne.concatenate_raws(raws=[data1_h, data2_h])
			self.raw_hf = mne.concatenate_raws(raws=[data1_hf, data2_hf])
			print(self.raw_h, self.raw_hf)

		elif is_raw_local is True: # In case data is stored locally
			self.raw_h, self.raw_hf = read_data(
        			type="raw", config=self.config, base_path=self.folder_path, verbose=verbose)
			if verbose is True:
				print("Loaded data:")
				print(self.raw_h, self.raw_hf)

		if is_event_local == True:
			self.events_h, self.events_hf = read_data(
				type="events", config=self.config, base_path=self.folder_path, verbose=verbose)
		else:
			self.events_h, _ = mne.events_from_annotations(self.raw_h, verbose=verbose)
			self.events_hf, _ = mne.events_from_annotations(self.raw_hf, verbose=verbose)

		if is_raw_filt_local == True and fast_start is False: # In case filtered data is stored locally
			self.raw_filt_h, self.raw_filt_hf = read_data(
				type="filtered", config=self.config, base_path=self.folder_path, verbose=verbose)
			if verbose is True:
				print(self.raw_filt_h, self.raw_filt_hf)

			self.IS_FILTERED = True

		if is_ica_local == True: # In case filtered data is stored locally
			self.ica_h, self.ica_hf = read_data(
				type="ica", config=self.config, base_path=self.folder_path, verbose=verbose)
			if verbose is True:
				print(self.ica_h, self.ica_hf)

			self.clean_raw_h, self.clean_raw_hf = read_data(
				type="clean", config=self.config, base_path=self.folder_path, verbose=verbose)
			if verbose is True:
				print(self.ica_h, self.ica_hf)

			self.IS_ICA = True

		if is_epoch_local is True:
			self.epochs = read_data(
				type="epochs", config=self.config, base_path=self.folder_path, verbose=verbose)
			if verbose is True:
				print(self.epochs)
			

	def get_raw(self):
		"""Returns EEG and noise data."""
		return self.raw_h, self.raw_hf

	def get_filt(self):
		"""Returns noise data."""
		return self.raw_filt_h, self.raw_filt_hf

	def get_ica(self):
		"""Returns noise data."""
		return self.ica_h, self.ica_hf

	def get_clean(self):
		"""Returns noise data."""
		return self.clean_raw_h, self.clean_raw_hf

	def get_events(self):
		"""Returns events data."""
		return self.events_h, self.events_hf

	def filter_data(self, freq_bands, tmax=None, verbose=False):
		"""Applies bandpass filter to EEG and noise data. Can also crop data."""
		if self.IS_FILTERED:
			raise(ValueError("Data has already been filtered..."))
		if self.raw_h is None or self.raw_hf is None:
			raise(ValueError("Data has not been loaded. Call `load_data()` before filtering."))
		if tmax:
			print(f"Cropping data to {tmax:.2f} seconds.")
			if isinstance(self.raw_h, mne.io.Raw):
				self.raw_h.crop(tmax=tmax)
			if isinstance(self.raw_hf, mne.io.Raw):
				self.raw_hf.crop(tmax=tmax)

		self.raw_filt_h = self.raw_h.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin", verbose=verbose)
		self.raw_filt_hf = self.raw_hf.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin", verbose=verbose)

		self.IS_FILTERED = True

	def plot_psd_ba_filt(self, freq_bands, verbose=False):
		"""Computes Power Spectral Density (PSD) for EEG and noise data
		before and after filtering."""

		# Plot the power spectral density (PSD) of the raw data
		fig = self.raw_h.compute_psd(picks=None).plot()

		if self.IS_FILTERED:
			title = 'PSD of concatenated Raw data after filtering'
		else:
			title = 'PSD of concatenated Raw data before filtering'
		fig.axes[0].set_title(title)
		plt.show()

		if self.IS_FILTERED:
			print("The data is already filtered, skipping filtering step.")
		else:
			self.filter_data(freq_bands, verbose=verbose)

		fig = self.raw_filt_h.compute_psd(picks=None).plot()
		fig.axes[0].set_title('PSD of concatenated Raw data after filtering')
		plt.show()

	def save_type_data(self, type, verbose=False):
		"""Saves cleaned data to a given filepath."""
		if type == "raw":
			save_data(self.raw_h, type, 1, self.config, self.folder_path, verbose=verbose)
			save_data(self.raw_hf, type, 2, self.config, self.folder_path, verbose=verbose)
		elif type == "events":
			save_data(self.events_h, type, 1, self.config, self.folder_path, verbose=verbose)
			save_data(self.events_hf, type, 2, self.config, self.folder_path, verbose=verbose)
		elif type == "filtered" and self.IS_FILTERED:
			save_data(self.raw_filt_h, type, 1, self.config, self.folder_path, verbose=verbose)
			save_data(self.raw_filt_hf, type, 2, self.config, self.folder_path, verbose=verbose)
		elif type == "ica" and self.IS_ICA:
			save_data(self.ica_h, type, 1, self.config, self.folder_path, verbose=verbose)
			save_data(self.ica_hf, type, 2, self.config, self.folder_path, verbose=verbose)
		elif type == "clean" and self.IS_ICA:
			save_data(self.clean_raw_h, type, 1, self.config, self.folder_path, verbose=verbose)
			save_data(self.clean_raw_hf, type, 2, self.config, self.folder_path, verbose=verbose)
		elif type == "epochs" and self.IS_ICA:
			save_data(self.epochs, type, 1, self.config, self.folder_path, verbose=verbose)
		else:
			raise ValueError("Data has not been proccessed correctly. Check the type and the data.")

	def decomp_ica(self, plt_show=False, n_components=None, verbose=False):
		if self.IS_FILTERED is False:
			raise ValueError("Data has not been filtered. Call `filter_data()` before applying ICA.")
		if self.IS_ICA is True:
			raise ValueError("Data has already been processed with ICA...")

		# Create and fit ICA for the first dataset
		self.ica_h = mne.preprocessing.ICA(n_components=n_components, random_state=42, method='fastica')
		self.ica_h.fit(self.raw_filt_h)

		# Create and fit ICA for the second dataset
		self.ica_hf = mne.preprocessing.ICA(n_components=n_components, random_state=42, method='fastica')
		self.ica_hf.fit(self.raw_filt_hf)

		self.ica_h = remove_eog(self.ica_h, self.raw_filt_h, plt=plt_show, title="EOG artifacts in individual Hands", verbose=False)
		self.ica_hf = remove_eog(self.ica_hf, self.raw_filt_hf, plt=plt_show, title="EOG artifacts in both Hands & Feet", verbose=False)

		self.ica_h = remove_ecg(self.ica_h, self.raw_filt_h, plt=plt_show, title="ECG artifacts in both Hands & Feet", verbose=False)
		self.ica_hf = remove_ecg(self.ica_hf, self.raw_filt_hf, plt=plt_show, title="ECG artifacts in both Hands & Feet", verbose=False)

		self.clean_raw_h = self.ica_h.apply(self.raw_filt_h.copy())
		self.clean_raw_hf =self.ica_hf.apply(self.raw_filt_hf.copy())

		self.IS_ICA = True

	def crt_epochs(self, data, events, event_dict, group_type, verbose=False):
		""" Cretates epochs to later apply CSP, ICA and feed the ML algorimth selected. """
		if self.IS_ICA is False:
			raise(ValueError("ICA has not been applied to the data. Please check the data."))
		if self.epochs is not None:
			print("Epochs have already been created... Re-creating epochs.")

		tmin = self.csp_config["tmin"]
		tmax = self.csp_config["tmax"]

		if group_type == 'hands':
			groupeve_dict = self.csp_config["event_dict_h"]
			freq_bands = self.csp_config["freq_exec_hands_02"]
		else:
			groupeve_dict = self.csp_config["event_dict_hf"]
			freq_bands = self.csp_config["freq_exec_hf"]

		event_dict = {key: value for key, value in groupeve_dict.items() if value in event_dict[0]}
		print("Event dict. : ", event_dict)

		self.epochs = mne.Epochs(data, events, event_dict, tmin=tmin, tmax=tmax, baseline=None, verbose=verbose)

		return self.epochs, freq_bands

	def csp(self, data, labels, freq_bands, epochs_info, verbose=False):
		tmin = self.csp_config["tmin"]
		tmax = self.csp_config["tmax"]

		n_components = self.csp_config["n_components"]
		fs = self.csp_config["frequency_sample"]

		features, csp = compute_csp(data, labels, freq_bands, n_components, fs, epochs_info, verbose=verbose)

		return features, csp

	def two_step_csp(self, epochs1, epochs2, freq_bands, verbose=False):
		""" Performs a two-step binary classification using CSP """
		from utils.csp import truncate_csp
		labels1 = epochs1.events[:, -1]
		labels2 = epochs2.events[:, -1]

		# **Step 1: Extract features for the first class**
		features_csp1, _ = self.csp(epochs1.get_data(), labels1, freq_bands, verbose=verbose)

		# **Step 2: Extract features for the second class**
		features_csp2 = self.csp(epochs2.get_data(), labels2, freq_bands, verbose=verbose)

		features_csp1, features_csp2, min_samples = truncate_csp(features_csp1, features_csp2)

		# **Final Step: Stack CSP1 and CSP2 features together**
		all_features = np.hstack([features_csp1, features_csp2])

		return all_features, labels2

	def csp_performance(self, epochs, labels, clf_type='lda', verbose=False):
		sfreq = self.csp_config["frequency_sample"]
		w_length = int(sfreq * 0.5)  # Running classifier: window length
		w_step = int(sfreq * 0.1)    # Running classifier: window step size
		w_start = np.arange(0, epochs.get_data().shape[2] - w_length, w_step)

		data = epochs.get_data()
		cv = ShuffleSplit(5, test_size=0.2, random_state=42)  # Keep ShuffleSplit for now

		# Define classifier
		if clf_type == 'lda':
			clf = LDA()
		elif clf_type == 'svm':
			clf = SVC(kernel='rbf', C=100, gamma=2, probability=False)
		elif clf_type == 'rf':
			clf = RandomForestClassifier(n_estimators=200, random_state=42)
		else:
			raise ValueError("Classifier type not recognized. Please use 'lda', 'svm' or 'rf'.")

		csp = CSP(n_components=self.csp_config["n_components"], reg='ledoit_wolf', log=True, norm_trace=False)  # CSP stays the same

		scores_windows = []

		for train_idx, test_idx in cv.split(data):
			# Split data
			X_train, X_test = data[train_idx], data[test_idx]
			y_train, y_test = labels[train_idx], labels[test_idx]


			# Fit CSP on the entire training set
			csp.fit(X_train, y_train)

			# Transform the entire dataset once
			X_train_csp = csp.transform(X_train)
			print(X_train_csp.shape)
			X_test_csp = csp.transform(X_test)

			X_train_csp = StandardScaler().fit_transform(X_train_csp)
			# Train classifier on full transformed training data
			clf.fit(X_train_csp, y_train)

			score_this_window = []

			for n in w_start:
				# Extract windowed segments from **already transformed** data
				X_test_win = X_test[:, :, n : n + w_length]

				# Apply the same CSP filters (already trained)
				X_test_csp_win = csp.transform(X_test_win)

				# Compute score on this window
				score_this_window.append(clf.score(X_test_csp_win, y_test))

			scores_windows.append(score_this_window)

		# Plot scores over time
		w_times = (w_start + w_length / 2.0) / sfreq + 0.3

		plt.figure()
		plt.plot(w_times, np.mean(scores_windows, axis=0), label="Score")
		plt.axvline(0, linestyle="--", color="k", label="Onset")
		plt.axhline(0.5, linestyle="-", color="k", label="Chance Level")
		plt.xlabel("Time (s)")
		plt.ylabel("Classification Accuracy")
		plt.title("Classification Score Over Time")
		plt.legend(loc="lower right")
		plt.show()

	def count_events(self, events, groupeve_dict, ev_dict):
		""" Extract discriminative features for binary classification tasks """

		# Filter event_dict to only keep specified event indices
		event_dict = {key: value for key, value in groupeve_dict.items() if value in ev_dict[0]}

		#* Count the number of events of each type
		for val, ev_name in zip(event_dict.values(), event_dict.items()):
			event_count = 0
			for event in events:	
				if event[2] == val:
					# print(event[2])
					event_count += 1
			print(f"Number of events of type {ev_name}: {event_count}")
		print()

	def normalize(self, epochs, verbose=False):
		# Initialize StandardScaler and PCA
		scaler = StandardScaler()

		# Standardize and apply PCA to each epoch individually
		std_epochs = np.empty((epochs.shape[0], epochs.shape[1], epochs.shape[2]))

		for i in range(epochs.shape[0]):
			epoch = epochs[i, :, :]  # Shape: (n_channels, n_times)
			epoch_std = scaler.fit_transform(epoch)  # Standardize
			std_epochs[i, :, :] = epoch_std

		# Verify the shape
		print("Shape after standardization:", std_epochs.shape)

		return std_epochs

	def pca(self, epochs, verbose=False):
		# Initialize StandardScaler and PCA
		pca = PCA(n_components=32)

		# Standardize and apply PCA to each epoch individually
		pca_epochs = np.empty((epochs.shape[0], 32, epochs.shape[2]))

		for i in range(epochs.shape[0]):
			epoch = epochs[i, :, :]  # Shape: (n_channels, n_times)
			epoch_pca = pca.fit_transform(epoch.T).T  # Apply PCA
			pca_epochs[i, :, :] = epoch_pca

		# Verify the shape
		print("Shape after PCA:", pca_epochs.shape)

		return pca_epochs

	def cross_val(self, X, y, pipeline, n_splits=5):
		# Evaluate the pipeline using cross-validation
		cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
		scores = cross_val_score(pipeline, X, y, cv=cv)

		print(scores)

		return scores

	def cross_validate(self, X, y, pipeline, n_splits=5):
		# Evaluate the pipeline using cross-validation
		cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
		scores = cross_validate(pipeline, X, y, cv=cv, return_train_score=True)
  
		return scores

	def train_model(self, X, y):

		self.lda = LDA(
			solver='lsqr',
			shrinkage='auto'
		)
		self.lda.fit(X, y)

		self.svm = SVC(
			kernel='rbf',
			C=200,
			gamma=0.1,
			probability=False
		)
		self.svm.fit(X, y)
		print(self.svm)

		self.rf = RandomForestClassifier(
			n_estimators=150,
			random_state=42
		)
		self.rf.fit(X, y)

		self.lgb_clf = lgb.LGBMClassifier(
			boosting_type='gbdt',  # Gradient Boosting Decision Trees
			n_estimators=50,      # Number of boosting rounds
			learning_rate=0.1,     # Step size shrinkage
			min_data_in_leaf=5,
			max_depth=-1,          # No depth limit
			num_leaves=31,         # Controls complexity
			random_state=42,
			verbosity=0
		)
		self.lgb_clf.fit(X, y)

		self.mlp = MLPClassifier(
			hidden_layer_sizes=(100, 50),
			max_iter=500
		)
		self.mlp.fit(X, y)

		print("Train Accuracy LDA:", accuracy_score(y, self.lda.predict(X)))
		print("Train Accuracy SVM:", accuracy_score(y, self.svm.predict(X)))
		print("Train Accuracy RF:", accuracy_score(y, self.rf.predict(X)))
		print("Train Accuracy RF:", accuracy_score(y, self.lgb_clf.predict(X)))
		print("Train Accuracy MLP:", accuracy_score(y, self.mlp.predict(X)))
		print()

	def save_models(self, X_test, y_test):

		if self.lda is None or self.svm is None or self.rf is None:
			raise(ValueError("The ML classification algorimths are not initialized. Call '.train_model()' before..."))

		# Save the model to a file
		save_model(self.folder_path, self.config, 'lda', self.lda)
		save_model(self.folder_path, self.config, 'svm', self.svm)
		save_model(self.folder_path, self.config, 'rf', self.rf)

		save_test_data(self.folder_path, self.config, X_test, y_test)

	def load_models(self):

		self.lda = load_model(self.folder_path, self.config, 'lda')
		self.svm = load_model(self.folder_path, self.config, 'svm')
		self.rf = load_model(self.folder_path, self.config, 'rf')
  
		print("All the ML classification algorimths have been correctly loaded.")
		print("You can now call '.pred()' to make predictions on the testing data.")

		X_test, y_test = load_test_data(self.folder_path, self.config) 
		return X_test, y_test


	def pred(self, X, y, n_preds=20, prt_matrix=False):
		if self.lda and self.svm and self.rf is None:
			raise ValueError("Model has not been trained. Call `train_model()` before predicting.")

		y_pred_lda = self.lda.predict(X)
		y_pred_svm = self.svm.predict(X)
		y_pred_rf = self.rf.predict(X)
		y_pred_lgb = self.lgb_clf.predict(X)
		y_pred_mlp = self.mlp.predict(X)

		print("epoch nb: [prediction] [truth] equal?")
		for i, (pred, real) in enumerate(zip(y_pred_svm, y)):
			is_correct = "True" if pred == real else "False"
			if i > n_preds:
				break
			print(f"epoch {i:03}:\t[{pred}]\t\t[{real}]  {is_correct}")
		print()

		print("LDA Accuracy:", accuracy_score(y, y_pred_lda))
		print("SVM Accuracy:", accuracy_score(y, y_pred_svm))
		print("Random Forest Accuracy:", accuracy_score(y, y_pred_rf))
		print("LGB Accuracy:\n", accuracy_score(y, y_pred_lgb))
		print("MLP Accuracy:\n", accuracy_score(y, y_pred_mlp))

		if prt_matrix:
			print("Confusion Matrix LDA:\n", confusion_matrix(y, y_pred_lda))
			print("Confusion Matrix SVM:\n", confusion_matrix(y, y_pred_svm))
			print("Confusion Matrix RF:\n", confusion_matrix(y, y_pred_rf))
			print("Confusion Matrix LGB:\n", confusion_matrix(y, y_pred_lgb))