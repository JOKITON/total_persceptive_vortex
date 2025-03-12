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
import colorama
from colorama import Fore, Style, Back

script_path = Path().resolve()
folder = (script_path / "../").resolve()

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

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

eeg_obj = EEGData(config_main, config_csp, folder, verbose=VERBOSE)

def predict():
    #* Can load the testing data along the ML models
    X_test, y_test = eeg_obj.load_models()
    eeg_obj.pred(X_test, y_test)

def train():
    data, _ = eeg_obj.get_clean()
    events, _ = eeg_obj.get_events()

    event_l = config_csp["ev_mlist_one"]
    groupeve_dict = config_csp["event_dict_h"]

    event_dict1 = {key: value for key, value in groupeve_dict.items() if value in event_l[0]}

    epochs = mne.Epochs(data, events, event_id=event_dict1, tmin=0.3, tmax=3.3, baseline=None, verbose=VERBOSE)
    data = epochs.get_data()
    # data = data.reshape(data.shape[0], -1)

    labels = epochs.events[:, -1]

    svm_clf = SVC(kernel='rbf', C=100, gamma=2, probability=True)
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

    ensemble = VotingClassifier(estimators=[('svm', svm_clf), ('rf', rf_clf)], voting='soft')

    # Create a new pipeline for LDA
    pipeline = Pipeline([
        ('csp', CSP(n_components=N_COMPONENTS_CSP, reg='ledoit_wolf', log=True, norm_trace=False)),
        ('scaler', StandardScaler()), #* StandardScaler works best
        ('pca', PCA(n_components=N_COMPONENTS_PCA)),
        ('voting_cs', ensemble)
    ])

    #* Split the data into train/test sets
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Fit the pipeline to the data
    pipeline.fit(data, labels)

    #* Transform the data using the pipeline
    # Create a new pipeline excluding the last step
    pipeline_without_last_step = Pipeline(pipeline.steps[:-1])

    # Transform the data using the new pipeline
    processed_data = pipeline_without_last_step.transform(data)

    #* Perform cross-validation
    scores = cross_val_score(pipeline, data, labels, cv=cv)

    print(scores)
    print("Cross_val_score:", np.mean(scores))
    print()

    #* Divide the data into training and testing sets ( a different way using the class )
    X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)

    eeg_obj.train_model(X_train, y_train)

    # eeg_obj.pred(X_test, y_test, n_preds=30)

def main():
    print("\n└─> Choose an option: (" + Fore.LIGHTWHITE_EX + Style.BRIGHT
                + "only numbers" + RESET_ALL + "): ")
    print("\t[1]" + Style.BRIGHT + " Train " + RESET_ALL + "the model." + RESET_ALL)
    print("\t[2] Make " + Style.BRIGHT + "predictions " + RESET_ALL + "with the model." + RESET_ALL)
    while 1:
        str1 = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        print(RESET_ALL)
        if str1.isdigit() is True:
            option = int(str1)

            if option == 1:
                train()
            elif option == 2:
                predict()
        else:
            print(
                Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
        continue

main()
