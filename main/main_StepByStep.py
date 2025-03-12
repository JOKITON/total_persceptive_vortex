import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact

# from conc_obj import EEGData
from classes.eeg_data import EEGData
from utils.plt import plot_psd, plot_montage
from utils.grid_search import grid_finder, grid_search
from utils.ica import plot_ica_comp
from utils.pipeline import crt_pipeline

# MNE imports
import mne
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import json

script_path = Path().resolve()
folder = (script_path / "../").resolve()

JSON_MAIN_PATH = script_path / "config/config_main.json"
JSON_CSP_PATH = script_path / "config/config_csp.json"
JSON_GRID_PATH = script_path / "config/config_grid.json"
EVENTS_PATH = script_path / "config/events.json"

with open(JSON_MAIN_PATH, "r") as f:
    config_main = json.load(f)

with open(JSON_CSP_PATH, "r") as f:
    config_csp = json.load(f)

VERBOSE = config_main['verbose'].lower() == 'true'

L_FREQ = config_main['l_freq']
H_FREQ = config_main['h_freq']

N_SUBJECTS = config_main["n_subjects"]
N_COMPONENTS_ICA = config_main["n_components_ica"]


eeg_obj = EEGData(config_main, config_csp, folder, verbose=VERBOSE)

#* Filters data and plots PSD to see differences
# eeg_obj.filter_data()
# eeg_obj.plot_psd_ba_filt(verbose=VERBOSE)

#* Can load the testing data along the ML models
# X_test, y_test = eeg_obj.load_models()
# eeg_obj.pred(X_test, y_test)

#* Plots different montages in 2D & 3D
# data = eeg_obj.get_raw_h()

# ch_names = data.info["ch_names"] 

# plot_montage(eeg_obj.montage, ch_names)

#* Computes ICA components ( If loaded locally do not use! )
# eeg_obj.decomp_ica(n_components=N_COMPONENTS_ICA, plt_show=True, verbose=VERBOSE)

#* Plot components of ICA
# plot_ica_comp(folder / config_main["path_ica_h"])

#* Loads cleaned data and events
data_h, data_hf = eeg_obj.get_clean()
events_h, events_hf = eeg_obj.get_events()

#* Creates epochs and frequency bands
ev_list = config_csp["ev_mlist_eight"]
epochs, freq_bands = eeg_obj.crt_epochs(data_h, events_h, ev_list, "hands", verbose=VERBOSE)

print()
epochs_data = epochs.get_data()
labels = epochs.events[:, -1]
print()

N_COMPONENTS_CSP = config_csp["n_components"]
features, csp = eeg_obj.csp(epochs_data, labels, freq_bands, epochs.info, verbose=VERBOSE)

#* Only use plot_patters if you are not using PCA before
# csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

#* Displays the performance of CSP along classifiers through a timeline
# eeg_obj.csp_performance(epochs, labels, clf_type='svm', verbose=False)

#* Two step CSP
# features, labels = eeg_obj.two_step_csp(epochs1, epochs2, freq_bands, verbose=VERBOSE)

#* Verify any shape
print("Shape after CSP:", features.shape)

#* Normalizes data
# features_norm = eeg_obj.normalize(features)
features_norm = StandardScaler().fit_transform(features)

#* Reduce dimensionality (PCA)
# features_pca = eeg_obj.pca(features_norm)
pca = PCA(n_components=N_COMPONENTS_CSP)
features_pca = pca.fit_transform(features_norm)

# grid = grid_finder(json_grid, 'svm', 'wide')
# print(grid)
# grid_search(data, labels, pipeline, grid)

pipeline = crt_pipeline(clf=True, voting='soft')

#* Trains and evaluates model
scores = eeg_obj.cross_val(features_pca, labels, pipeline, n_splits=5)
print("Mean score:", np.mean(scores))

scores = eeg_obj.cross_validate(features_pca, labels, pipeline, n_splits=5)

#* Divide the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, random_state=42, shuffle=True)

eeg_obj.train_model(X_train, y_train)

eeg_obj.pred(X_test, y_test, n_preds=30)

#* Saves filtered and concatenated data for faster loading
# eeg_obj.save_type_data(type="events", folder_path=folder, verbose=VERBOSE)
# eeg_obj.save_type_data(type="raw")
# eeg_obj.save_type_data(type="filtered")
# eeg_obj.save_type_data(type="ica")
# eeg_obj.save_type_data(type="clean")
# eeg_obj.save_type_data(type="epochs")
# eeg_obj.save_models()
