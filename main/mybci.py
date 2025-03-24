import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact

# from conc_obj import EEGData
from obj.eeg_data import EEGData
from utils.plt import plot_psd, plot_montage
from utils.ica import plot_ica_comp

# MNE imports
import mne
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
# from mne.decoding import CSP
from csp.CSPObj_cheat import CSP as FBCSP

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
processed_X_train, y_train, processed_X_test, y_test = None, None, None, None

def predict():
    global processed_X_test
    global y_test

    #* Can load the testing data along the ML models
    eeg_obj.pred(processed_X_test, y_test, n_preds=20)

def train():
    global processed_X_test
    global y_test

    data, _ = eeg_obj.get_clean()
    events, _ = eeg_obj.get_events()

    event_l = config_csp["ev_blist_one"]
    groupeve_dict = config_csp["event_dict_h"]

    event_dict1 = {key: value for key, value in groupeve_dict.items() if value in event_l[0]}

    epochs = mne.Epochs(data, events, event_id=event_dict1, tmin=0.3, tmax=3.3, baseline=None, verbose=VERBOSE)
    data = epochs.get_data()
    # data = data.reshape(data.shape[0], -1)

    labels = epochs.events[:, -1]

    #* Divide the data into training and testing sets ( a different way using the class )
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    freq_bands = config_csp["freq_bands__02"]
    # Create a new pipeline for LDA
    pipeline = Pipeline([
        ('csp', FBCSP(n_components=config_csp["n_components"], freq_bands=freq_bands, fs=config_csp["frequency_sample"], log=True, norm_trace=False)),
        ('scaler', StandardScaler()), #* StandardScaler works best
        ('pca', PCA(n_components=0.99)),
        ('mlp', MLPClassifier(hidden_layer_sizes=[128, 256, 256], max_iter=1000, learning_rate_init=0.001, alpha=0.0005, random_state=42, learning_rate='adaptive'))
    ])

    #* Split the data into train/test sets
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Fit the pipeline to the data
    pipeline.fit(X_train, y_train)

    #* Perform cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=cv)

    print(scores)
    print("Cross_val_score:", np.mean(scores))
    print()

    # Create a new pipeline excluding the last step
    pipeline_without_last_step = Pipeline(pipeline.steps[:-1])

    # Transform the data using the new pipeline
    processed_X_train = pipeline_without_last_step.transform(X_train)
    processed_X_test = pipeline_without_last_step.transform(X_test)
    print(processed_X_train.shape)

    eeg_obj.train_model(processed_X_train, y_train)

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
