import operator
import numpy as np
from scipy.signal import firwin


class BSA:
    def __init__(self, threshold=0.955, num_taps=20, cutoff=0.8, fs=None):
        self.threshold = threshold
        self.num_taps = num_taps
        self.cutoff = cutoff
        self.fs = fs
        # Spikes will be populated after calling 'encode_spikes'
        self.spikes = None

    @property
    def _fir_filter(self):
        if self.fs is not None:
            fir_coefficients = firwin(self.num_taps, cutoff=self.cutoff, fs=self.fs)
        else:
            fir_coefficients = firwin(self.num_taps, cutoff=self.cutoff)
        fir = {"fir_coeffs": fir_coefficients, "n_taps": len(fir_coefficients)}
        return fir

    def encode_spikes(self, data, normalize=True):
        n_channels = data.shape[0]
        n_samples = data.shape[1]
        fir_filter = self._fir_filter

        # Set the threshold vaules
        if isinstance(self.threshold, float) or isinstance(self.threshold, int):
            bsa_threshold = self.threshold
        else:
            raise TypeError("Please provide a single threshold value.")

        # Normalise data to be between 0 and 1
        # NOTE: the original algorithm has been designed for values between 0 and 1
        if normalize:
            min_data = np.min(data, axis=1)
            # Move waves to positive values
            positive_data = data - min_data[:, np.newaxis]
            # Normalize between 0 and 1
            max_data = np.max(data, axis=1)
            encoding_data = positive_data / (max_data - min_data)[:, np.newaxis]
        else:
            encoding_data = data
        # Initialize the output
        output = np.zeros([n_channels, n_samples], dtype=np.int8)
        # BSA Algorithm
        for _channel in range(n_channels):
            for _sample in range(n_samples):
                error1 = 0
                error2 = 0
                # Compute the err. 1 as the difference between the actual data and filter response
                for j in range(fir_filter["n_taps"]):
                    if _sample + j - 1 <= n_samples:
                        error1 += abs(
                            encoding_data[_channel][_sample]
                            - fir_filter["fir_coeffs"][j]
                        )
                        # Compute err. 2 as the cumulative amplitude of the signal in the filter window
                        error2 += abs(encoding_data[_channel][_sample])
                # If the prediction error is lower than the signal amplitude super threshold, emit a spike
                if error1 <= (error2 - bsa_threshold):
                    output[_channel, _sample] = 1
                    # If a spike has been emitted, lower the subsequent signal values
                    for j in range(fir_filter["n_taps"]):
                        if _sample + j - 1 <= n_samples:
                            encoding_data[_channel][_sample] -= fir_filter[
                                "fir_coeffs"
                            ][j]

        self.spikes = output

        return output

    def decode_eeg(self, channel=1):
        # Utility function to decode the timeserie represented by the spikes
        encoded_signal = self.spikes
        sliding_filter = self._fir_filter["fir_coeffs"]

        return np.convolve(encoded_signal[channel, :], sliding_filter, "same")
