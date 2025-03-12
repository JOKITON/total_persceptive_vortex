import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path

class EEGData:
    def __init__(self, eeg_path, noise_path, l_freq, h_freq):
        """Initialize EEGData object with file paths and optional filter settings."""
        self.eeg_path = eeg_path
        self.noise_path = noise_path
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.load_data()

    def load_data(self):
        """Loads EEG and noise data."""
        self.raw_eeg = mne.io.read_raw_fif(self.eeg_path, preload=True, verbose=False)
        self.raw_noise = mne.io.read_raw_fif(self.noise_path, preload=True, verbose=False)

    def filter_data(self):
        """Applies bandpass filter to EEG and noise data."""
        self.raw_eeg_filtered = self.raw_eeg.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin")
        self.raw_noise_filtered = self.raw_noise.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin")

    def compute_psd(self):
        """Computes Power Spectral Density (PSD) for EEG and noise data."""
        data_eeg = self.raw_eeg_filtered.get_data()
        data_noise = self.raw_noise_filtered.get_data()
        sfreq = self.raw_eeg.info["sfreq"]

        self.psd_eeg, self.freqs = mne.time_frequency.psd_array_welch(data_eeg, fmin=self.l_freq, fmax=self.h_freq, sfreq=sfreq)
        self.psd_noise, _ = mne.time_frequency.psd_array_welch(data_noise, fmin=self.l_freq, fmax=self.h_freq, sfreq=sfreq)

        self.psd_eeg_mean = np.mean(self.psd_eeg, axis=0)
        self.psd_noise_mean = np.mean(self.psd_noise, axis=0)

        # Convert to dB
        #? Add a small value to avoid log(0)
        self.psd_eeg_db = 10 * np.log10(self.psd_eeg_mean + 1e-10)
        self.psd_noise_db = 10 * np.log10(self.psd_noise_mean + 1e-10)

    def plot_psd(self):
        """Plots EEG and noise PSD."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.freqs, self.psd_eeg_db, label="EEG/MEG Data", color="blue")
        plt.plot(self.freqs, self.psd_noise_db, label="Empty Room Noise", color="red", linestyle="dashed")

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB)")
        plt.title("PSD Comparison: EEG vs. Noise")
        plt.legend()
        plt.grid()
        plt.show()
