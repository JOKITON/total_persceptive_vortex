import mne
import numpy as np
import mne
from mne.io import read_raw_edf
from mne.datasets import eegbci
import pickle

def get_path(base_path, config, type, subtype):
	"""Saves raw data to a given filepath."""

	path = None
	if type == "raw":
		if subtype == 1:
			path = base_path / config["path_raw_h"]
		elif subtype == 2:
			path = base_path / config["path_raw_hf"]
	elif type == "filtered":
		if subtype == 1:
			path = base_path / config["path_filt_h"]
		elif subtype == 2:
			path = base_path / config["path_filt_hf"]
	elif type == "ica":
		if subtype == 1:
			path = base_path / config["path_ica_h"]
		elif subtype == 2:
			path = base_path / config["path_ica_hf"]
	elif type == "clean":
		if subtype == 1:
			path = base_path / config["path_clean_h"]
		elif subtype == 2:
			path = base_path / config["path_clean_hf"]
	elif type == "epochs":
		path = base_path / config["path_epochs"]
	elif type == "events":
		if subtype == 1:
			path = base_path / config["path_events_h"]
		elif subtype == 2:
			path = base_path / config["path_events_hf"]
	elif type == "events_exec":
		if subtype == 1:
			path = base_path / config["path_events_exec_h"]
		elif subtype == 2:
			path = base_path / config["path_events_exec_hf"]
	elif type == "events_img":
		if subtype == 1:
			path = base_path / config["path_events_img_h"]
		elif subtype == 2:
			path = base_path / config["path_events_img_hf"]
	elif type == "models":
		if subtype == 1:
			path = base_path / config["path_lda"]
		elif subtype == 2:
			path = base_path / config["path_svm"]
		elif subtype == 3:
			path = base_path / config["path_rf"]
	else:
		raise ValueError(f"Invalid type {type}.")

	return path

def save_train_data(base_path, config, X_test, y_test):
	path_X = base_path / config["path_X_train"]
	path_y = base_path / config["path_y_train"]

	np.save(path_X, X_test)
	np.save(path_y, y_test)

def save_test_data(base_path, config, X_test, y_test):
	path_X = base_path / config["path_X_test"]
	path_y = base_path / config["path_y_test"]

	np.save(path_X, X_test)
	np.save(path_y, y_test)

def load_test_data(base_path, config):
	path_X = base_path / config["path_X_test"]
	path_y = base_path / config["path_y_test"]

	X_test = np.load(path_X)
	y_test = np.load(path_y)

	return X_test, y_test

def save_model(base_path, config, subtype, alg):

	path = None
	if subtype == "lda":
		path = get_path(base_path, config, "models", 1)
	elif subtype == "svm":
		path = get_path(base_path, config, "models", 2)
	elif subtype == "rf":
		path = get_path(base_path, config, "models", 3)
	else:
		raise(ValueError("The given classification algorimth does not exist in the project. Try with another one..."))

	if alg is not None and path is not None:
		with open(path, 'wb') as file:
			pickle.dump(alg, file)

def load_model(base_path, config, subtype):

	path = None
	if subtype == "lda":
		path = get_path(base_path, config, "models", 1)
	elif subtype == "svm":
		path = get_path(base_path, config, "models", 2)
	elif subtype == "rf":
		path = get_path(base_path, config, "models", 3)
	else:
		raise(ValueError("The given classification algorimth does not exist in the project. Try with another one..."))

	if path is not None:
		with open(path, 'rb') as file:
			ret_alg = pickle.load(file)
		return ret_alg

def read_data(type, config, base_path, verbose=False):
	path_h = get_path(base_path, config, type, 1)
	path_hf = get_path(base_path, config, type, 2)

	if type == 'events_exec' or type == 'events_img' or type == 'events':
		data_h = mne.read_events(path_h)
		data_hf = mne.read_events(path_hf)
	elif type == 'ica':
		data_h = mne.preprocessing.read_ica(path_h, verbose=verbose)
		data_hf = mne.preprocessing.read_ica(path_hf, verbose=verbose)
	elif type != "ica":
		data_h = mne.io.read_raw_fif(path_h, preload=True, verbose=verbose)
		data_hf = mne.io.read_raw_fif(path_hf, preload=True, verbose=verbose)
	else:
		raise ValueError(f"Invalid type. {type}")

	return data_h, data_hf

def fetch_data(subject, runs, event_dict, montage, verbose=False):
	"""Loads EEG data from files and appends to a list."""
	raw_temp = []
	for p in subject:
		for i in runs:
			data_path = str(eegbci.load_data(p, i)[0])
			# Read the data and store Raw objects in a list
			raw = read_raw_edf(data_path, preload=True, stim_channel='auto', verbose=verbose)

			# Standarize channel pos and names
			eegbci.standardize(raw)

			# Create a montage object and set it to the raw object
			raw.set_montage(montage)

			# Load the events from the annotations
			events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1,T1=2,T2=3), verbose=verbose)
			
			annot_from_events = mne.annotations_from_events(
				events=events, event_desc=event_dict, sfreq=raw.info['sfreq'],
				orig_time=raw.info['meas_date'], verbose=verbose)
			raw.set_annotations(annot_from_events)

			# Append the concatenated Raw object to the list
			raw_temp.append(raw)

	raw_temp = mne.concatenate_raws(raw_temp)
	all_events, _ = mne.events_from_annotations(raw_temp, verbose=verbose)
	return raw_temp, all_events

def save_data(data, type, subtype, config, base_path, verbose=False):
	if subtype:
		path = get_path(base_path, config, type, subtype)

	# In case events are to be saved as numpy arrays
	if type == "events_exec" or type == "events_img" or type == "events":
		mne.write_events(path, data, overwrite=True)
		return
	elif type == "epochs":
		data.save(path, overwrite=True, verbose=verbose)
		return
	data.save(path, overwrite=True, verbose=verbose)
