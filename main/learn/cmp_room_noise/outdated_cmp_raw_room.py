""" Main project """

import numpy as np
from pathlib import Path
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

# Get the sample data, if there isn't any, it fetches automatically
script_path = Path(__file__).resolve().parent
sample_data_folder = (script_path / "../sample_data").resolve()

sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
)

sample_data_raw_noise = (
    sample_data_folder / "MEG" / "sample" / "ernoise_raw.fif"
)

# Load the raw data
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=True)

# Load empty-room noise recording
raw_noise = mne.io.read_raw_fif(sample_data_raw_noise, preload=True)

# Get info of raw data
info = raw.info

# Get array of raw data
data = raw.get_data()
data_noise = raw_noise.get_data()

# Compute PSD manually
psd_raw, freqs = mne.time_frequency.psd_array_welch(data, fmax=50, sfreq=raw.info["sfreq"])
psd_noise, _ = mne.time_frequency.psd_array_welch(data_noise, fmax=50, sfreq=raw_noise.info["sfreq"])

# Average across all channels
psd_raw_mean = np.mean(psd_raw, axis=0)
psd_noise_mean = np.mean(psd_noise, axis=0)

# Convert power to dB
psd_raw_mean_db = 10 * np.log10(psd_raw_mean)
psd_noise_mean_db = 10 * np.log10(psd_noise_mean)
psd_diff_db = 10 * np.log10(psd_raw_mean / psd_noise_mean)

# Plot the two PSDs
plt.figure(figsize=(10, 6))
plt.plot(freqs, psd_raw_mean_db, label="EEG/MEG Data", color="blue")
plt.plot(freqs, psd_noise_mean_db, label="Empty Room Noise", color="red", linestyle="dashed")
plt.plot(freqs, psd_diff_db, label="EEG/MEG - Noise", color="green")

# Formatting
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB)")
plt.title("PSD Comparison: EEG/MEG Data vs. Empty Room Noise")
plt.legend()
plt.grid()
plt.show()
