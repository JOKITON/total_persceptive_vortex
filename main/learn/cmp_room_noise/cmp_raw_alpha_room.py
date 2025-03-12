""" Plots the alpha band (8-12 Hz) power spectral density (PSD) of EEG/MEG data and empty room noise. """

import numpy as np
from pathlib import Path
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from egg_model import EEGData

def	plot_raw_room_noise(l_freq, h_freq):

	# Get the sample data,
	script_path = Path(__file__).resolve().parent
	sample_data_folder = (script_path / "../sample_data").resolve()

	sample_data_raw_file = (
		sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
	)

	sample_data_raw_noise = (
		sample_data_folder / "MEG" / "sample" / "ernoise_raw.fif"
	)

	eeg_data = EEGData(sample_data_raw_file, sample_data_raw_noise, l_freq=l_freq, h_freq=h_freq)
	eeg_data.filter_data()
	eeg_data.compute_psd()
	eeg_data.plot_psd()

plot_raw_room_noise(6, 14)
