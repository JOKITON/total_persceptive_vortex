import matplotlib.pyplot as plt
import mne

def plot_psd(data, title, verbose=False):
		# Plot the power spectral density (PSD) of the raw data
		fig = data.compute_psd(picks=None).plot()

		fig.axes[0].set_title(title)
		plt.show()

def plot_montage(add_montage, ch_names):
		# Print all the available montages
		# print(mne.channels.get_builtin_montages())

		# The system used to record the data is BioSemi ActiveTwo, which has 64 channels
		biosemi_montage = mne.channels.make_standard_montage('biosemi64')

		# 2D montage plot
		fig = plt.figure()
		biosemi_montage.plot(kind='topomap', show_names=True)
		plt.show()

		# 3D montage plot
		fig = plt.figure()
		add_montage.plot(kind='3d', show_names=ch_names)
		plt.show()
