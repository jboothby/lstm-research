import pandas as pd
import numpy as np
import ecgdetectors
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import normalize

"""
Take the collaborator ECG data and preprocess it
    1.) Read the signal as CSV
    2.) Normalize signal by dividing by max value
    3.) run peak detection on signal
    4.) split signal into 1 second windows around peaks
    5.) filter signal with bandpass filter
"""


class DataProcessor:
    def __init__(self, window_size, high_cut, low_cut, order, fs):
        self.WINDOW_SIZE = window_size
        self.HIGHCUT = high_cut
        self.LOWCUT = low_cut
        self.ORDER = order
        self.FS = fs

    def butter_filter(self, signal):
        nyquist = self.FS / 2
        low = self.LOWCUT / nyquist
        high = self.HIGHCUT / nyquist
        b, a = butter(self.ORDER, [low, high], 'band')
        return filtfilt(b, a, signal)

    def preprocess(self, freq, filename):
        # Read csv into signalframe, extract signal
        df = pd.read_csv(filename)
        signal = df['Signal'].tolist()

        # plt.figure(figsize=(12,8))
        # plt.title(f'Kemal {freq}hz raw signal downsampled to 360hz')
        # plt.plot(np.linspace(0,1,freq), signal[:freq], '-r', label='raw')

        # resample the data to change the frequency
        time = pd.timedelta_range(0, periods=len(signal), freq=f"{(1 / freq):.8f}S")
        df = pd.DataFrame(index=time, data={'signal': signal})
        df = df.resample(f'{1 / self.FS:.8f}S').mean()  # resample frequency to 360hz
        signal = df['signal']

        # plt.plot(np.linspace(0,1,FS), signal[:FS], '-b', label='downsampled')
        # plt.legend()
        # plt.show()

        # scale all signal values to be between 0 and 1
        max_signal_value = max(signal)
        signal = [point / max_signal_value for point in signal]

        # create peak detection object, initialize sampling frequency
        detectors = ecgdetectors.Detectors(self.FS)
        r_peaks = detectors.engzee_detector(signal)

        # get signal value of each peak in peak detection and create coordinates
        annotation_signal = []
        for x in range(len(r_peaks)):
            annotation_signal.append(signal[r_peaks[x]])
        annotation_coords = [[x, y] for x, y in zip(r_peaks, annotation_signal)]

        return signal, annotation_coords

    def graph_signal(self, signal, annotation_coords):
        """Graph the signal with the peaks"""
        plt.figure(figsize=(12, 8))

        # set up pyplot graph for given seconds
        x = np.arange(5 * self.FS)
        y = signal[:5 * self.FS]

        # label the graph
        plt.title(f"EKG Data, Kemal 360Hz ")
        plt.xlabel("Sample")
        plt.ylabel("Reading[mV]")
        plt.ylim(np.min(y) - 0.1, np.max(y) + 0.4)  # set y limits to allow for annotations
        plt.plot(x, y, '-b', label='Signal')  # plot points from first lead (T is transpose)
        for a in annotation_coords[:4]:
            # annotate 1/10 second forward and 0.2 mV above point, with arrow to point
            plt.annotate('Peak', xy=(a[0], a[1]), xytext=(a[0] + (self.FS / 10), a[1] + 0.1),
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
        plt.legend()
        plt.show()

    def create_windows(self, signal, annotation_coords):

        windows = []
        # create windows
        for a in annotation_coords:
            # skip signals that are too close to the end of signal
            if abs(a[0] - len(signal)) < self.WINDOW_SIZE:
                continue

            if self.WINDOW_SIZE == 360:
                window = signal[a[0] - self.WINDOW_SIZE // 2: a[0] + self.WINDOW_SIZE // 2]
            elif self.WINDOW_SIZE == 216:
                window = signal[a[0] - self.WINDOW_SIZE // 3: a[0] + (2 * (self.WINDOW_SIZE // 3))]
            else:
                print("Program not configured for this window size")
                exit(0)
            windows.append(window)

        return windows
