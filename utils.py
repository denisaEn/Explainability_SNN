import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from autoreject import AutoReject


class EegData:
    """Utlity class to read EEG data and perform pre-processing"""

    def __init__(self, eeg_data, len_segments=None, overlap=None, preload_data=True):
        epochs = []
        if len_segments is not None or overlap is not None:
            epochs = mne.make_fixed_length_epochs(
                eeg_data, duration=len_segments, overlap=overlap, preload=preload_data
            )
        self._data = eeg_data
        self.ar = AutoReject()
        self.epochs = epochs

    @property
    def data(self):
        return self._data.get_data()

    @property
    def shape(self):
        """Return the shape of the timeseries data,
        i.e. C x N x M, with C = num. of intervals, N = num. of channels,
        M = num. of temporal samples per interval

        :return: Shape of the data array
        :rtype: tuple
        """
        return self.data.shape

    @classmethod
    def from_mat_file(cls, filename, data_key=None, header_info={}, *args, **kwargs):
        """Initialize a new EegData object from a .mat file


        :param filename: the path to the desired .mat file
        :type filename: string
        :param data_key: the key used to access the data matrix in the '.mat' file
        :type data_key: string
        :param header_info: data acquisition information (e.g., number/list of channels and sampling frequency)
        :type header_info: dict
        :return: an EegData object storing the data contained in the file
        :rtype: EegData
        """

        file_data = scipy.io.loadmat(filename)
        # The mat file could be a dict with misc header information
        if data_key is not None:
            data = file_data[data_key]
        # NOTE: numpy data needs to be converted in a mne.io.RawArray object,
        # which requires basic information about the data
        channels = header_info.get("channels", 32)
        s_frequency = header_info.get("s_frequency", 128)

        eeg_info = mne.create_info(channels, s_frequency)
        raw_data = mne.io.RawArray(data, eeg_info)
        
        #selected_channels = [ 'FT9', 'O1', 'FC6', 'Fp2', 'Oz','F4', 'T8', 'C3']
        # .locs filepath: "/content/drive/MyDrive/eeg_dataset/Coordinates.xls"
        locs_info_path = "/content/drive/MyDrive/eeg_dataset/Coordinates.locs"
        # import channel location information
        montage = mne.channels.read_custom_montage(locs_info_path)
        # import correct channel names
        new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
        print(new_chan_names)
        # get old channel names
        old_chan_names = raw_data.info["ch_names"]
        # create a dictionary to match old channel names and new (correct) channel names
        chan_names_dict = {old_chan_names[i]:new_chan_names[i] for i in range(32)}
        # update the channel names in the dataset
        raw_data.rename_channels(chan_names_dict)
        # check location information
        raw_data.set_montage(montage)
        #drop_channels = [channel for channel in new_chan_names if channel not in selected_channels]
        #raw_data.drop_channels(drop_channels)
        #raw_data.describe()
        #print(raw_data.info["ch_names"])
        return EegData(raw_data, *args, **kwargs)

    @classmethod
    def from_set_file(cls, filename, *args, **kwargs):
        """Initialize a new EegData object from a .set file


        :param filename: the path to the desired .set file
        :type filename: string
        :return: an EegData object storing the data contained in the file
        :rtype: EegData
        """
        raw_data = mne.io.read_raw_eeglab(filename)
        return EegData(raw_data, *args, **kwargs)

    def filter(self, *args, low_cut_f=0.5, high_cut_f=50.0, in_place=True, **kwargs):
        filtered_data = self._data.filter(
            l_freq=low_cut_f, h_freq=high_cut_f, *args, **kwargs
        )
        if in_place:
            self._data = filtered_data
        else:
            return filtered_data

    def reject_artifacts(self, in_place=True):
        try:
            clean_data = self.ar.fit_transform(self._data)
        except Exception as e:
            print(
                f"Make sure to provide segmented raw data, instantiated with the `preload` flag set to `True`! Error: {e}"
            )
        if in_place:
            self._data = clean_data
        else:
            return clean_data

    def plot(self, channels=None, time=None, cmap_name="viridis"):
        from matplotlib.cm import get_cmap

        if channels is None:
            print("Please provide the indices of the desired channels to plot!")
        else:
            cmap = get_cmap(cmap_name)
            norm_values = np.linspace(0, 1, len(channels))
            _, ax = plt.subplots(len(channels), 1, sharex=True)
            for i, (ch, norm) in enumerate(zip(channels, norm_values)):
                if isinstance(time, int):
                    ax[i].plot(
                        np.array(range(time)),
                        self.data[ch, :time],
                        c=cmap(norm),
                    )
                elif isinstance(time, (tuple, list)):
                    ax[i].plot(
                        time[0] + np.array(range(time[1] - time[0])),
                        self.data[ch, time[0] : time[1]],
                        c=cmap(norm),
                    )
                else:
                    ax[i].plot(
                        np.array(range(self.shape[1])), self.data[ch, :], c=cmap(norm)
                    )
            plt.show()
