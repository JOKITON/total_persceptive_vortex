import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
from mne.io import RawArray
from mne import create_info

class EEGData:
    def __init__(self, eeg_path, noise_path, l_freq=None, h_freq=None, verbose=False):
        """Initialize EEGData object with file paths and optional filter settings."""
        self.eeg_path = eeg_path
        self.noise_path = noise_path

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.load_data(verbose=verbose)

    def load_data(self, verbose=False):
        """Loads EEG and noise data."""
        self.raw_eeg = mne.io.read_raw_fif(self.eeg_path, preload=True, verbose=verbose)
        # Cut down duration to 60s
        # self.raw_eeg.crop(tmax=60.0).pick(picks=["mag", "eeg", "stim", "eog"])

        # Extract events directly from the EEG raw data
        self.events = mne.find_events(self.raw_eeg, verbose=verbose)

        self.raw_noise = mne.io.read_raw_fif(self.noise_path, preload=True, verbose=verbose)
    
    def load_event_dict(self, event_json):
        """Saves event dictionary from JSON."""

        # Convert keys from string to int (JSON stores keys as strings)
        self.event_dict = event_json
    
    def get_raw_data(self):
        """Returns EEG and noise data."""
        return self.raw_eeg

    def get_raw_noise(self):
        """Returns noise data."""
        return self.raw_noise

    def get_events(self):
        """Returns events data."""
        return self.events, self.event_dict
    
    def apply_pca(self, n_components=None):
        """Applies PCA to EEG data and reduces dimensions."""
        tmin, tmax = -0.1, 0.3
        picks = mne.pick_types(
            self.raw_eeg_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        epochs = mne.Epochs(
            self.raw_eeg_filtered,
            self.events,
            self.event_dict,
            tmin,
            tmax,
            proj=False,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False,
        )
        pca = UnsupervisedSpatialFilter(PCA(30), average=False)
        pca_data = pca.fit_transform(epochs)
        ev = mne.EvokedArray(
            np.mean(pca_data, axis=0),
            mne.create_info(30, epochs.info["sfreq"], ch_types="eeg"),
            tmin=tmin,
        )
        ev.plot(show=False, window_title="PCA", time_unit="s")
        exit()

        pca = PCA(n_components=n_components)
        data = self.raw_eeg_filtered.get_data().T
        pca.fit(data)
        if n_components is None:
            # Plot explained variance to choose n_components
            plt.figure(figsize=(10, 6))
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Explained Variance by Number of Principal Components')
            plt.show()

            return pca
        else:
            print(data.shape)
            self.reduced_data = pca.transform(data)
            print(self.reduced_data.shape)
            info = create_info(ch_names=[f'PC{i+1}' for i in range(n_components)], sfreq=self.raw_eeg.info['sfreq'])
            info.set_montage('standard_1020')  # Set a standard montage
            for ch in info['chs']:
                ch['kind'] = 2  # Set channel type to EEG (2 is the code for EEG in MNE)
            self.reduced_raw = RawArray(self.reduced_data.T, info)

    def filter_data(self, tmax=None, verbose=False):
        """Applies bandpass filter to EEG and noise data. Can also crop data."""
        if tmax:
            print(f"Cropping data to {tmax:.2f} seconds.")
            self.raw_eeg.crop(tmax=tmax)
            self.raw_noise.crop(tmax=tmax)
        self.raw_eeg_filtered = self.raw_eeg.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin", verbose=verbose)
        self.raw_noise_filtered = self.raw_noise.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin", verbose=verbose)

    def compute_ica(self, n_comp, plot_comp=False, plot_arts=False, verbose=False):
        if not hasattr(self, "reduced_raw"):
            raise ValueError("PCA-reduced data not found. Call `apply_pca()` before running ICA.")

        ica = ICA(n_components=n_comp, random_state=97, max_iter=800)
        ica.fit(self.reduced_raw)  # Fit ICA on raw EEG data
        
        if plot_comp is True:
            # Plot ICA components
            ica.plot_components()

        # Detect ECG artifacts
        ecg_inds, ecg_scores = ica.find_bads_ecg(inst=self.reduced_raw, method='correlation', verbose=verbose)
        # Detect EOG artifacts
        eog_inds, eog_scores = ica.find_bads_eog(inst=self.reduced_raw, ch_name='EOG 061', verbose=verbose)

        if plot_arts is True:
            
            ica.plot_scores(eog_scores)
            print("\nPlotting scores of EOG components...")
            ica.plot_scores(ecg_scores)
            print("\nPlotting scores of ECG components...")

            # Visualize the identified components
            if ecg_inds:
                # Flatten lists in case they are nested
                ecg_inds = [comp for sublist in ecg_inds for comp in sublist] if ecg_inds and isinstance(ecg_inds[0], list) else ecg_inds
                ica.plot_sources(self.reduced_data, picks=ecg_inds)
            if eog_inds:
                ica.plot_sources(self.reduced_data, picks=eog_inds)
                eog_inds = [comp for sublist in eog_inds for comp in sublist] if eog_inds and isinstance(eog_inds[0], list) else eog_inds

        plt.show()

        # Exclude the identified components
        ica.exclude = ecg_inds + eog_inds
        print(f"Excluding ICA components: {ica.exclude}")

        # Apply ICA only if there are components to exclude
        if ica.exclude:
            self.raw_clean = ica.apply(self.reduced_raw.copy(), verbose=verbose)  # Apply ICA on a copy
        else:
            print("No components to exclude, skipping ICA application.")
            self.raw_clean = self.reduced_raw.copy()

        return self.raw_clean  # Return cleaned data

    def compute_psd(self, verbose=False):
        """Computes Power Spectral Density (PSD) for EEG and noise data."""
        data_eeg = self.reduced_raw.get_data()
        data_noise = self.reduced_raw.get_data()
        sfreq = self.raw_eeg.info["sfreq"]

        self.psd_eeg, self.freqs = mne.time_frequency.psd_array_welch(data_eeg, fmin=self.l_freq, fmax=self.h_freq, sfreq=sfreq, verbose=verbose)
        self.psd_noise, _ = mne.time_frequency.psd_array_welch(data_noise, fmin=self.l_freq, fmax=self.h_freq, sfreq=sfreq, verbose=verbose)

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
    
    def plot_clean_eeg(self, title="Cleaned EEG Data"):
        """Plots cleaned EEG data."""
        self.raw_clean.plot(title=title)
        plt.show()

    def plot_evoked_events(self, event_str1, event_str2, type="eeg", verbose=False):
        ncols = 1
        event_keys = []
        for key, value in self.event_dict.items():
            if key == event_str1 or key == event_str2:
                event_keys.append(value - 1)  # Append only event name

        # Pick only EEG channels
        if type == "eeg": # EEG is measured using uV (microVolts)
            ncols = 1
            eeg_channels = mne.pick_types(self.raw_clean.info, eeg=True)
        elif type == "meg": # MEG is measured using Teslas (fT/cm)
            ncols = 2
            eeg_channels = mne.pick_types(self.raw_clean.info, meg=True)

        epochs = mne.Epochs(
            self.raw_clean, self.events, event_id=event_keys,  # Now using correct event IDs
            tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True,
            verbose=verbose, picks=eeg_channels
        )

        evoked_epochs_one = epochs[event_keys[0]].average()
        evoked_epochs_two = epochs[event_keys[1]].average()

        # Create a single figure with multiple subplots
        fig, axes = plt.subplots(1, ncols, figsize=(22, 10))
        fig = mne.viz.plot_compare_evokeds(
            [evoked_epochs_one, evoked_epochs_two],
            title=type.upper(),
            axes=axes,
            show=False,
            legend=False
        )
        if ncols == 1:
            axes.legend([event_str1, event_str2], loc="upper right")
            axes.set_ylabel("uV (Microvolts)", rotation="horizontal", labelpad=40)
        elif ncols == 2:
            axes[1].legend([event_str1, event_str2], loc="upper right")
            axes[0].set_ylabel("fT/cm (Teslas)", rotation="horizontal", labelpad=20)
            axes[1].set_ylabel("")

        plt.show()
