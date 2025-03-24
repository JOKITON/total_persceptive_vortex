import numpy as np

import mne
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.csp import compute_csp
from utils.data import read_data, save_data, fetch_data
from utils.ica import remove_eog

class EEGDataAuto:

	def __init__(self, config, config_csp, folder, event_dict, freq_bands, verbose=False):
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
		self.epochs : mne.Epochs = None
		self.lda, self.svm, self.rf = None, None, None

		# Initialize basic variables to fill later
		self.folder_path = folder
		self.config = config
		self.csp_config = config_csp

		self.subject = np.arange(config["n_subjects"]) + 1

		# Create a montage object to store the channel positions
		self.montage = mne.channels.make_standard_montage(config["montage"])

		# We will need event dict later on for event specific Epochs
		self.event_dict = event_dict
		self.freq_bands : list = freq_bands

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

		# Filter data
		if self.raw_filt_h is None or self.raw_filt_hf is None:
			self.filter_data()

		# Decompose EEG using ICA
		if self.clean_raw_h is None or self.clean_raw_hf is None:
			self.decomp_ica(n_components=config["n_components_ica"])

		# Create Epochs
		if self.epochs is None:
			# Create Epochs
			if self.csp_config["group_type"] == "hands":
				data = self.clean_raw_h
				events = self.events_h
			else:
				data = self.clean_raw_hf
				events = self.events_hf
			self.crt_epochs(data, events, verbose=verbose)

		self.features = self.epochs.get_data()
		self.labels = self.epochs.events[:, -1]

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

	def get_features(self):
		"""Returns features."""
		return self.features

	def get_labels(self):
		"""Returns labels."""
		return self.labels

	def get_events(self):
		"""Returns events data."""
		return self.events_h, self.events_hf

	def filter_data(self, tmax=None, verbose=False):
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
		self.ica_h = mne.preprocessing.ICA(n_components=n_components, random_state=42, method='fastica', verbose=verbose)
		self.ica_h.fit(self.raw_filt_h)

		# Create and fit ICA for the second dataset
		self.ica_hf = mne.preprocessing.ICA(n_components=n_components, random_state=42, method='fastica', verbose=verbose)
		self.ica_hf.fit(self.raw_filt_hf)

		self.ica_h = remove_eog(self.ica_h, self.raw_filt_h, plt=plt_show, title="EOG artifacts in individual Hands", verbose=verbose)
		self.ica_hf = remove_eog(self.ica_hf, self.raw_filt_hf, plt=plt_show, title="EOG artifacts in both Hands & Feet", verbose=verbose)

		self.clean_raw_h = self.ica_h.apply(self.raw_filt_h.copy(), verbose=verbose)
		self.clean_raw_hf =self.ica_hf.apply(self.raw_filt_hf.copy(), verbose=verbose)

		self.IS_ICA = True

	def crt_epochs(self, data, events, verbose=False):
		""" Cretates epochs to later apply CSP, ICA and feed the ML algorimth selected. """
		if self.IS_ICA is False:
			raise(ValueError("ICA has not been applied to the data. Please check the data."))
		if self.epochs is not None:
			print("Epochs have already been created... Re-creating epochs.")

		tmin = self.csp_config["tmin"]
		tmax = self.csp_config["tmax"]

		print("Event dict. : ", self.event_dict)

		self.epochs = mne.Epochs(data, events, self.event_dict, tmin=tmin, tmax=tmax, baseline=None, verbose=verbose)

	def csp(self, data, labels, verbose=False):
		tmin = self.csp_config["tmin"]
		tmax = self.csp_config["tmax"]

		n_components = self.csp_config["n_components"]
		fs = self.csp_config["frequency_sample"]

		features, csp = compute_csp(data, labels, self.freq_bands, n_components, fs, self.epochs.info, verbose=verbose)

		return features, csp

	def normalize(self, epochs, verbose=False):
		# Initialize StandardScaler
		scaler = StandardScaler()

		# Standardize each epoch individually
		std_epochs = np.empty((epochs.shape[0], epochs.shape[1], epochs.shape[2]))

		for i in range(epochs.shape[0]):
			epoch = epochs[i, :, :]  # Shape: (n_channels, n_times)
			epoch_std = scaler.fit_transform(epoch)  # Standardize
			std_epochs[i, :, :] = epoch_std

		# Verify the shape
		print("Shape after standardization:", std_epochs.shape)

		return std_epochs

	def pca(self, epochs, verbose=False):
		# Initialize PCA
		pca = PCA(n_components=32)

		# Apply PCA to each epoch individually
		pca_epochs = np.empty((epochs.shape[0], 32, epochs.shape[2]))

		for i in range(epochs.shape[0]):
			epoch = epochs[i, :, :]  # Shape: (n_channels, n_times)
			epoch_pca = pca.fit_transform(epoch.T).T  # Apply PCA
			pca_epochs[i, :, :] = epoch_pca

		# Verify the shape
		print("Shape after PCA:", pca_epochs.shape)

		return pca_epochs
