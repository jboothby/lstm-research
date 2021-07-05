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

WINDOW_SIZE = 216       # size of window for split
FS = 360                # sampling frequency
HIGHCUT = 40            # filter parameters
LOWCUT = 0.5
ORDER = 5


def butter_filter(signal, lowcut=LOWCUT, highcut=HIGHCUT, order=ORDER):
    nyquist = FS / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], 'band')
    return filtfilt(b, a, signal)

def normalize(data_point):
    return data_point / MAX_SIGNAL_VALUE

# Read csv into signalframe, extract signal
df = pd.read_csv('Kemal360Hz.csv')
signal = df['Signal'].tolist()

# scale all signal values to be between 0 and 1
MAX_SIGNAL_VALUE = max(signal)
signal = list(map(normalize, signal))

# create peak detection object, initialize sampling frequency
detectors = ecgdetectors.Detectors(FS)
r_peaks = detectors.engzee_detector(signal)

# get signal value of each peak in peak detection and create coordinates
annotation_signal = []
for x in range(len(r_peaks)):
    annotation_signal.append(signal[r_peaks[x]])
annotation_coords = [[x,y] for x,y in zip(r_peaks, annotation_signal)]

print(annotation_coords)

"""Graph the signal with the peaks"""
plt.figure(figsize=(12,8))
# set up pyplot graph for given seconds
x = np.arange(1500)[400:]
y = signal[400:1500]

# label the graph
plt.title(f"EKG Data, Kemal 360Hz ")
plt.xlabel("Sample")
plt.ylabel("Reading[mV]")
plt.ylim(np.min(y) - 0.1, np.max(y)+0.4)                # set y limits to allow for annotations
plt.plot(x, y, '-b', label='Signal')  # plot points from first lead (T is transpose)
for a in annotation_coords[:4]:
    # annotate 1/10 second forward and 0.2 mV above point, with arrow to point
    plt.annotate('Peak', xy=(a[0], a[1]), xytext=(a[0]+(FS/10), a[1] + 0.1),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
plt.legend()
plt.show()

windows = []
# create windows
for a in annotation_coords:
    if WINDOW_SIZE == 360:
        window = signal[a[0] - WINDOW_SIZE // 2: a[0] + WINDOW_SIZE // 2]
    elif WINDOW_SIZE == 216:
        window = signal[a[0] - WINDOW_SIZE // 3: a[0] + (2 * (WINDOW_SIZE // 3))]
    else:
        print("Program not configured for this window size")
        exit(0)

    windows.append(window)
# filter window signals
filtered = list(map(butter_filter, windows))

"""Plot the raw vs filtered ECG signal"""
plt.title("Raw vs. Filtered ECG Signal")
plt.plot(windows[0], '-b', label='raw')
plt.plot(filtered[0], 'r', label='filtered')
plt.legend()
plt.show()
