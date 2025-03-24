import numpy as np
import scipy.signal as signal

def frequency_perturbation(epoch_data, shift_hz, sfreq):
    """
    Apply frequency perturbation by shifting the spectrum.
    
    Parameters:
    - epoch_data: numpy array of shape (n_channels, n_times)
    - shift_hz: frequency shift in Hz (e.g., 0.5 Hz)
    - sfreq: sampling frequency of the EEG data
    
    Returns:
    - perturbed_epoch: numpy array of the same shape as epoch_data
    """
    n_channels, n_times = epoch_data.shape
    perturbed_epoch = np.zeros_like(epoch_data)
    
    for ch in range(n_channels):
        # Apply FFT
        freqs = np.fft.rfftfreq(n_times, d=1/sfreq)
        fft_vals = np.fft.rfft(epoch_data[ch])
        
        # Shift the spectrum
        shift_samples = int(shift_hz * n_times / sfreq)
        fft_vals = np.roll(fft_vals, shift_samples)
        
        # Apply inverse FFT
        perturbed_epoch[ch] = np.fft.irfft(fft_vals, n=n_times)
    
    return perturbed_epoch
