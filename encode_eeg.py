import numpy as np
import matplotlib.pyplot as plt
from utils_all_channels import EegData
from bsa import BSA
from poisson import poisson
from scipy.stats import pearsonr

from scipy import signal

# NOTE: the .mat files contain matrices of shape (32,3200)
# with 32 = num of channels, 3200 = 25 sec of recodring with 128 samples per second (SPS)


def plot_encoded_eeg(data, spikes, bsa, channel=1, rescale=False):
    fig, axs = plt.subplots(2)

    # Get a single channel as example
    eeg_row = data[channel, :]
    n_samples = data.shape[1]
    spikes_row = spikes[channel, :]

    # Decode the spikes
    decoded_spikes = bsa.decode_eeg(channel)
    # Move and scale the EEG signal
    # Either rescale the reconstructed EEG to the original domain
    if rescale:
        min_data = np.min(data, axis=1)
        max_data = np.max(data, axis=1)
        # Scale the decoded signal from peak to peak
        amplitude = max_data[channel]  # - min_data[channel]
        scaled_decoded = decoded_spikes * amplitude
        # Move waves to positive values
        decoded_eeg = scaled_decoded - abs(min_data[channel])
        axs[0].plot(np.array(range(n_samples)), eeg_row)
        axs[0].plot(np.array(range(n_samples)), decoded_eeg)
    # Or plot the normalised original data used for encoding
    else:
        positive_eeg = eeg_row - np.min(eeg_row)
        normalsed_eeg = positive_eeg / (np.max(eeg_row) - np.min(eeg_row))
        axs[0].plot(np.array(range(n_samples)), normalsed_eeg)
        axs[0].plot(np.array(range(n_samples)), decoded_spikes)

    axs[0].margins(0, 0.05)
    # axs[1].set_axis_off()
    axs[1].imshow(
        spikes_row.reshape(1, -1), cmap="binary", aspect="auto", interpolation="nearest"
    )
    # Finally, display the plot
    plt.show()


# Read a test EEG from the figshare dataset
eeg = EegData.from_mat_file(
    "eeg_dataset/filtered_data/Arithmetic_sub_20_trial1.mat",
    data_key="Clean_data",
)
print(eeg.data.dtype)

# Run the encoding algorithm and plot the result
bsa = BSA(threshold=0.100, num_taps=24, cutoff=63, fs=128)
spikes = bsa.encode_spikes(eeg.data, normalize=False)
#plot_encoded_eeg(eeg.data, spikes, bsa)

# Select a single channel from the EEG data
single_channel_data = eeg.data[1, :]
# Normalize the data
max_data = np.max(single_channel_data)
min_data = np.min(single_channel_data)
normalized_data = (single_channel_data - min_data) / (max_data - min_data)
# Encode the EEG data using the Poisson method
poisson_spikes = poisson.encode(normalized_data, num_timesteps=2560)
poisson_spikes_lowtime = poisson.encode(normalized_data, num_timesteps=128)

# Decode the spikes
decoded_poisson_spikes_lowtime = poisson.decode(poisson_spikes_lowtime)* (max_data - min_data) + min_data
decoded_poisson_spikes = poisson.decode(poisson_spikes)* (max_data - min_data) + min_data
decoded_bsa_spikes = bsa.decode_eeg(1)

# Apply the pearsonr()
corr_poisson, _ = pearsonr(eeg.data[1, :], decoded_poisson_spikes)
print('Pearsons correlation for poisson with num_timesteps=2560: %.3f' % corr_poisson)
corr_poisson_low, _ = pearsonr(eeg.data[1, :], decoded_poisson_spikes_lowtime)
print('Pearsons correlation for poisson with num_timesteps=128: %.3f' % corr_poisson_low)
corr_bsa, _ = pearsonr(eeg.data[1, :], decoded_bsa_spikes)
print('Pearsons correlation for bsa: %.3f' % corr_bsa)

# Compute the power spectra using FFT for each signal
fft1 = np.fft.fft(eeg.data[1, :])
fft2 = np.fft.fft(decoded_bsa_spikes)
fft3 = np.fft.fft(decoded_poisson_spikes)
fft4 = np.fft.fft(decoded_poisson_spikes_lowtime)

# Compute the power spectral density for each signal
psd1 = np.abs(fft1) ** 2
psd2 = np.abs(fft2) ** 2
psd3 = np.abs(fft3) ** 2
psd4 = np.abs(fft4) ** 2

# Compute the frequencies for each bin
freqs = np.fft.fftfreq(len(eeg.data[1, :]), 1/128)

# Plot the power spectra for the original, BSA-encoded, and Poisson-encoded signals
plt.semilogy(freqs[:len(freqs)//2], psd1[:len(psd1)//2], label='Original signal', alpha=0.7)
plt.semilogy(freqs[:len(freqs)//2], psd2[:len(psd2)//2], label='BSA-encoded signal', alpha=0.7)
plt.semilogy(freqs[:len(freqs)//2], psd3[:len(psd3)//2], label='Poisson-encoded signal ts=2560', alpha=0.7)
plt.semilogy(freqs[:len(freqs)//2], psd4[:len(psd4)//2], label='Poisson-encoded signal ts=128', alpha=0.7)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (μV^2/Hz)')
plt.grid()
plt.legend()
plt.show()