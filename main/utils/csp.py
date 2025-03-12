import numpy as np
from scipy.signal import butter, filtfilt
import mne
from mne.decoding import CSP
# from csp.CSPObj import CSP
 
def butter_bandpass(lowcut, highcut, fs, order=5):
		""" Creates a bandpass Butter filter """
		nyq = 0.5 * fs  # Frecuencia de Nyquist
		low = lowcut / nyq
		high = highcut / nyq
		b, a = butter(order, [low, high], btype='band')
		return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
	""" Applies a bandpass filter to the data """
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	return filtfilt(b, a, data, axis=-1)
  
def compute_csp(epochs_data, labels, freq_bands, n_components, fs, epochs_info=None, verbose=False):
	""" Extract discriminative features for binary classification tasks """

	all_features = []
	print(epochs_data.shape, labels.shape)
	for lowcut, highcut in freq_bands:
		# Filtrar los datos en la banda seleccionada
		filtered_data = np.array([
			apply_bandpass_filter(epoch, lowcut, highcut, fs) for epoch in epochs_data
		])

		# Aplicar CSP en la banda filtrada
		csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, norm_trace=False)
		features = csp.fit_transform(filtered_data, labels)

		if epochs_info is not None:
			csp.plot_patterns(epochs_info, ch_type='eeg', units='uV', size=1.5)

		all_features.append(features)

	# Concatenar características de todas las bandas
	all_features = np.concatenate(all_features, axis=1)

	return all_features, csp

def truncate_csp(csp_one, csp_two):
	min_samples = min(csp_one.shape[0], csp_two.shape[0])

	features_csp1 = csp_one[:min_samples]
	features_csp2 = csp_two[:min_samples]

	return features_csp1, features_csp2, min_samples
