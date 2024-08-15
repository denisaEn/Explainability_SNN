import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt

class poisson:
    def encode(data, num_timesteps, scale=1):
        """
        Encode a flattened array into a Poisson event series.

        Parameters:
        - data: numpy array representing the input data (flattened)
        - num_timesteps: integer representing the number of time steps
        - scale: scaling factor for the Poisson process (default: 0.1)

        Returns:
        - encoded_data: numpy array representing the encoded event series
        """

        # Reshape data to 2D array with single row
        data = data.reshape(1, -1)

        # Repeat data across time dimension
        data = np.repeat(data, num_timesteps, axis=0)

        # Generate Poisson events
        poisson_values = np.random.rand(*data.shape)
        encoded_data = (poisson_values < (data * scale)).astype(float)
        
        return encoded_data

    def decode(encoded_data, scale=1, input_range=(0, 1)):
        """
        Decode an encoded Poisson event series to approximate the original values.

        Parameters:
        - encoded_data: numpy array representing the encoded event series
        - scale: scaling factor used for encoding (default: 0.1)
        - input_range: tuple representing the range of the original input values (default: (0, 1))

        Returns:
        - decoded_data: numpy array representing the decoded values
        """

        # Sum the event counts across time steps
        event_counts = np.sum(encoded_data, axis=0)

        # Decode by dividing event counts by the scale factor
        decoded_data = event_counts / scale

        # Scale decoded data to the original input range
        decoded_data = np.interp(decoded_data, (0, encoded_data.shape[0]), input_range)

        return decoded_data