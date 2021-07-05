import wfdb
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt, lfilter
from matplotlib import pyplot as plt
import os

db_path = os.path.join(os.getcwd(), 'mit_database')


def read_file(patnum, begin=0, end=5):
    # patnum is filename w/o extension
    record_path = os.path.join(db_path, str(patnum))

    # show read record method
    """
    record = wfdb.rdrecord(record_path)
    attributes = vars(record)
    print(attributes)

    print('-' * 50 )
    """

    # show read sample method
    signals, fields = wfdb.rdsamp(record_path)
    print(f"Signals and Fields\n{signals}\n{fields}")

    freq = fields['fs']  # get frequency from fields, used to determine sample point coords
    start = int(begin * freq)
    stop = int(end * freq)


    print('-' * 50)

    """ Demonstrates annotation read ( must include extension type) """
    """

    ann = wfdb.rdann(record_path, 'atr')                            # read annotation file
    ann_atts = vars(ann)                                            # convert object to dict
    samples = np.array(ann_atts['sample'])                          # extract sample array

    # grab annotations for the correct range of seconds
    annotations = np.fromiter((x for x in samples if start <= x and x <= stop), dtype=samples.dtype)
    first_index = np.where(samples == annotations[0])[0][0]
    last_index = first_index + len(annotations)

    # get corresponding symbols
    annotation_text = ann_atts['symbol'][first_index : last_index]
    print(f"{samples}\n{annotations}\n{annotation_text}")

    # build lookup dict for annotation symbols
    ann_dict = {}
    for a, b in zip(annotations, annotation_text):
        ann_dict[a] = b

    annotation_coords = [[x, y] for x in annotations for y in signals[x]]   # create x,y coords for annotation points
    print(f"Annotation coordinates: {annotation_coords}")

    figure, axis = plt.subplots(3,1)

    plt.figure(figsize=(12,8))


    # set up pyplot graph for given seconds
    x = np.arange(start + 1, stop + 1)
    y = signals[start:stop]
    # label the graph
    #axis[0].title(f"EKG Data, Patient {patnum} ")
    #axis[0].xlabel("Sample")
    #axis[0].ylabel("Reading[mV]")
    #axis[0].ylim(np.min(y) - 0.1, np.max(y)+0.5)                # set y limits to allow for annotations
    axis[0].plot(x, y.T[0], '-b', label=fields['sig_name'][0])  # plot points from first lead (T is transpose)
    axis[0].plot(x, y.T[1], '-r', label=fields['sig_name'][1])  # plot points from second lead
    for a in annotation_coords:
        # annotate 1/10 second forward and 0.2 mV above point, with arrow to point
        axis[0].annotate(ann_dict[a[0]], xy=(a[0], a[1]), xytext=(a[0]+(freq/10), a[1] + 0.2),
                 arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
    #axis[0].legend()
    """

    # perform fft on signal
    # fourier(pd.DataFrame(signals.T[0][start:stop].to_numpy()), int(freq))
    butter_filter(pd.DataFrame(signals.T[0])[start:stop].to_numpy().flatten(), 25, 360, 3)
    print(fields)

def butter_filter(signal, critical, frequency, order=5):
    # b is numerator, a is denominator of polynomials of IIR filter
    nyquist = frequency / 2
    low = critical / nyquist
    b, a = butter(order, low)
    y = lfilter(b, a, signal.T)


    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_figwidth(20)
    fig.set_figheight(8)
    ax1.plot(np.arange(0, len(signal), 1), signal)
    ax2.plot(np.arange(0, len(signal), 1), y)

    plt.xlabel("Data point")
    plt.ylabel("Voltage Reading [mV]")
    fig.suptitle(f"Butterworth lowpass filter for smoothing ECG data, Patient 101")
    plt.xticks(ticks=None)
    plt.show()


"""
    # second part analysis
    nobs = len(signal_tensor)
    signal_ft = np.abs(np.fft.rfft(signal_tensor))
    signal_freq = np.fft.rfftfreq(nobs, d=1 / freq)

    x = signal_freq
    y = signal_ft[:901]
    axis[2].plot(x, y)

    def annot_max(x, y, ax=None):
        xmax = x[np.argmax(y)]  # argmax returns indices of max y values
        ymax = y.max()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrow_props = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrow_props, bbox=bbox_props, ha="right", va="top")
        if not ax:
            ax = plt.gca()
        text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)

    annot_max(x, y, axis[2])

    plt.xlabel('frequency (1/samp)')
    """


read_file(102, begin=3, end=5)
