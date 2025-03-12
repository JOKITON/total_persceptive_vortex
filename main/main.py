import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact

# from conc_obj import EEGData
from classes.eeg_data import EEGData
from utils.plt import plot_psd, plot_montage
from utils.ica import plot_ica_comp

# MNE imports
import mne
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
# from mne.decoding import CSP
from csp.CSPObj import CSP

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
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
    
with open(JSON_GRID_PATH, "r") as f:
    json_grid = json.load(f)

VERBOSE = config_main['verbose'].lower() == 'true'

L_FREQ = config_main['l_freq']
H_FREQ = config_main['h_freq']

N_SUBJECTS = config_main["n_subjects"]
N_COMPONENTS_ICA = config_main["n_components_ica"]
N_COMPONENTS_CSP = config_csp["n_components"]
N_COMPONENTS_PCA = N_COMPONENTS_CSP

"""
T0 corresponds to rest
T1 corresponds to onset of motion (real or imagined) of
the left fist (in runs 3, 4, 7, 8, 11, and 12)
both fists (in runs 5, 6, 9, 10, 13, and 14)
T2 corresponds to onset of motion (real or imagined) of
the right fist (in runs 3, 4, 7, 8, 11, and 12)
both feet (in runs 5, 6, 9, 10, 13, and 14)
"""

eeg_obj = EEGData(config_main, config_csp, folder, verbose=VERBOSE)

#* Filters data and plots PSD to see differences
# eeg_obj.filter_data()
# eeg_obj.plot_psd_ba_filt(verbose=VERBOSE)

# eeg_obj.plot_psd(verbose=VERBOSE)

#* Normalizes data
# eeg_obj.normalize_data()

#* Can load the testing data along the ML models
# X_test, y_test = eeg_obj.load_models()
# eeg_obj.pred(X_test, y_test)

#* Plots different montages in 2D & 3D
# data = eeg_obj.get_raw_h()
# ch_names = data.info["ch_names"] 
# plot_montage(eeg_obj.montage, ch_names)

#* Computes ICA components
# eeg_obj.decomp_ica(n_components=N_COMPONENTS_ICA, plt_show=True, verbose=VERBOSE)

#* Plot components of ICA
# plot_ica_comp(folder / config["path_ica_h"])

data, _ = eeg_obj.get_clean()
events, _ = eeg_obj.get_events()

event_l = config_csp["ev_mlist_eight"]
groupeve_dict = config_csp["event_dict_h"]

event_dict1 = {key: value for key, value in groupeve_dict.items() if value in event_l[0]}
print("Event dict. :", event_dict1)

print()
epochs = mne.Epochs(data, events, event_id=event_dict1, tmin=0.3, tmax=3.3, baseline=None, verbose=VERBOSE)
data = epochs.get_data()
# data = data.reshape(data.shape[0], -1)

labels = epochs.events[:, -1]
print("X:", data.shape)
print("Y:", labels.shape)

svm_clf = SVC(kernel='rbf', C=100, gamma=2, probability=True)
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

ensemble = VotingClassifier(estimators=[('svm', svm_clf), ('rf', rf_clf)], voting='soft')

# Create a new pipeline for LDA
pipeline = Pipeline([
    #* Does not work (it needs reshaping)
    # ('ica', FastICA(n_components=64)),

    ('csp', CSP(n_components=N_COMPONENTS_CSP, reg='ledoit_wolf', log=True, norm_trace=False)),

    ('scaler', StandardScaler()), #* StandardScaler works best
    # ('scaler', MinMaxScaler()),
    # ('scaler', RobustScaler()),

    ('pca', PCA(n_components=N_COMPONENTS_PCA)),

	('voting_cs', ensemble)
    #* Choose only one of the following classifiers
	# ('lda', LDA())
    # ('svm', SVC(kernel='rbf', C=100, gamma=2))
	# ('rf', RandomForestClassifier())
])

# Fit the 3D pipeline to the transformed data
# pipeline.fit(data, labels)

from utils.grid_search import grid_finder, grid_search

#* Perform GridSearch to find the optimal parameters with different ML algorimths
# grid = grid_finder(json_grid, 'svm', 'wide')
# print(grid)
# grid_search(data, labels, pipeline, grid)

#* Split the data into train/test sets
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Fit the pipeline to the data
pipeline.fit(data, labels)

# Transform the data using the pipeline
# Create a new pipeline excluding the last step
pipeline_without_last_step = Pipeline(pipeline.steps[:-1])

# Transform the data using the new pipeline
processed_data = pipeline_without_last_step.transform(data)
print(processed_data.shape)

#* Perform cross-validation
scores = cross_val_score(pipeline, data, labels, cv=cv)

print("Cross-validation scores:", scores)
print("Processed data shape:", processed_data.shape)

#* Divide the data into training and testing sets ( a different way using the class )
X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)

eeg_obj.train_model(X_train, y_train)

eeg_obj.pred(X_test, y_test, n_preds=30)

#* Saves filtered and concatenated data for faster loading
# eeg_obj.save_type_data(type="raw")
# eeg_obj.save_type_data(type="filtered")
# eeg_obj.save_type_data(type="norm")
# eeg_obj.save_type_data(type="ica")
# eeg_obj.save_models()
