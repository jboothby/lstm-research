# import tensorflow_core.python.keras.models
import wfdb
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import Callback
import os
import csv
import warnings
from time import perf_counter
from sklearn.model_selection import train_test_split as Split
from sklearn.metrics import classification_report
import time
import ecgdetectors
from Benchmark import load_binary_testing_data

tf.enable_v2_behavior()

# set up pandas df display options to view full set in console
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

DIRECTORY_NAME = 'mit_database'

# hyperparameters
WINDOW_SIZE = 216
WINDOW_STEP = 1  # Take every nth point in the window. So step 4 is every 4th point
EPOCHS = 50
VERBOSE = 2
DENSE_LAYERS = 45
LSTM_LAYERS = 23
BATCH_SIZE = 32  # size of each batch
BATCH_LOG_SIZE = 300  # number of batches to take place before logging loss/accuracy
HIGHCUT = 40  # allow frequencies below this rate (filters 50/60hz main noise)
LOWCUT = 0.5  # allow frequencies above this rate (filters 0.05-0.5 hz muscle noise due to harmonics)
ORDER = 5  # order of butterworth filter
DATA_SAMPLE_RATE = 360
FS = 360

CLASSIFICATION = {'N': 0, 'L': 1, 'R': 2, 'B': 3, 'A': 4, 'a': 5, 'J': 6, 'S': 7, 'V': 8,
                  'r': 9, 'F': 10, 'e': 11, 'j': 12, 'n': 13, 'E': 14, '/': 15, 'f': 16, 'Z': 17}

# CLASSIFICATION = {'N': 0, 'V': 1, 'F': 2, 'E': 3, 'Q': 4, 'O': 5, 'P': 6, 'Z': 7}

""" Mapping of Physionet to AHA Codes is as follows:
[., N, L, R, A, a, J, s, e, j] -> N
[V] -> V
[F, f] -> F
[!, p] -> O
[E] -> E
[/, P] -> P
[Q] -> Q
Everything else -> Z
"""

METRICS = [
    keras.metrics.CategoricalAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn')
]


# read MIT database into training, validating, testing data
def parse_datafiles():
    print('Parsing data files...')
    start = time.time()
    filenums = []
    data = []
    labels = []

    # Get all filenames from arrythmia database folder
    for filename in os.listdir(os.path.join(os.getcwd(), '../mit_database')):
        base = os.path.splitext(filename)[0]
        # add files that are new
        if base not in filenums:
            filenums.append(base)

    # get information for each patient, load into dataframe
    for filenum in filenums:

        # open and read each file
        record_path = os.path.join(os.getcwd(), '../mit_database', str(filenum))
        signals, fields = wfdb.rdsamp(record_path)
        ann = vars(wfdb.rdann(record_path, 'atr'))

        if 'MLII' not in fields['sig_name']:  # skip entries with no MLII signal for consistency
            continue

        # get signal from MLII lead and smooth with butterworth low pass
        MLII_signal = signals.T[fields['sig_name'].index('MLII')]

        # iterate over annotations, generate window, insert into proper data group
        for i in range(int(ann['ann_len'])):
            index = int(ann['sample'][i])

            label = ann['symbol'][i]

            # skip annotations that are closer than a window width to the edge
            if index < WINDOW_SIZE or (fields['sig_len'] - index) < WINDOW_SIZE:
                continue

            # set slicing properly based on window size
            # training and validation data get smoothed windows, training is unsmoothed and not stepped yet
            if WINDOW_SIZE == 360:
                window = MLII_signal[index - WINDOW_SIZE // 2: index + WINDOW_SIZE // 2: WINDOW_STEP]
            elif WINDOW_SIZE == 216:
                # for 600 ms window, below.
                window = MLII_signal[index - (WINDOW_SIZE // 3): index + ((WINDOW_SIZE * 2) // 3): WINDOW_STEP]
            else:
                print(f"Program not configured for WINDOW_SIZE: {WINDOW_SIZE}")
                exit(0)

            data.append(window)
            labels.append(label)

    # make lists into properly shaped numpy arrays
    labels = np.array(labels).reshape(-1, 1)
    data = np.array(data).reshape(-1, WINDOW_SIZE, 1)

    # put info into dataframe so it can be stratified
    df_data = pd.DataFrame(data.reshape(data.shape[0], data.shape[1]), index=np.arange(data.shape[0]))
    df_labels = pd.DataFrame(labels, index=np.arange(data.shape[0]))
    df = pd.concat([df_labels, df_data], axis=1)

    # split and stratify data
    training_set, testing_set = Split(df, stratify=df.iloc[:, 0])
    training_set, validation_set = Split(training_set, stratify=training_set.iloc[:, 0])

    # split sets into labels and data, return as tuples
    train_label = label_array_to_oncehot(training_set.iloc[:, 0])
    train_data = training_set.iloc[:, 1:].to_numpy().reshape(-1, WINDOW_SIZE, 1)

    validate_label = label_array_to_oncehot(validation_set.iloc[:, 0])
    validate_data = validation_set.iloc[:, 1:].to_numpy().reshape(-1, WINDOW_SIZE, 1)

    test_label = label_array_to_oncehot(testing_set.iloc[:, 0])
    test_data = testing_set.iloc[:, 1:].to_numpy().reshape(-1, WINDOW_SIZE, 1)

    stop = time.time()
    print(f'Finished parsing data files in {stop - start} seconds')
    return (train_data, train_label), (validate_data, validate_label), (test_data, test_label)


def label_array_to_oncehot(arr):
    np_arr = np.array(list(map(annotation_map, arr))).reshape(-1, 1)
    return tf.keras.utils.to_categorical(np_arr, num_classes=len(CLASSIFICATION))


def annotation_map(n):
    if n in list(CLASSIFICATION.keys()):
        n = n
    else:
        n = 'Z'
    return CLASSIFICATION[n]


def butter_filter(signal, lowcut, highcut, order=3):
    nyquist = DATA_SAMPLE_RATE / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], 'band')
    return filtfilt(b, a, signal)


# class allows for outputting loss per batch instead of per epoch
# shamelessly stole this from https://github.com/keras-team/keras/issues/2850
# added the bit that saves batch losses and batch accuracy for more granular plotting
class NBatchLogger(Callback):
    def __init__(self, display=10):
        # display is number of batches to wait before outputting loss
        self.step = 0
        self.metric_cache = {}
        self.display = display
        self.batch_losses = []
        self.batch_accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            self.batch_losses.append(self.metric_cache['loss'] / self.display)
            self.batch_accuracy.append(self.metric_cache['accuracy'] / self.display)
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step * BATCH_SIZE % self.params['samples'],
                                              self.params['samples'],
                                              metrics_log))
            self.metric_cache.clear()


class Data:
    train, validate, test = parse_datafiles()

    def __init__(self):
        """ Original model using dense layer with lstm """
        self.model = tf.keras.Sequential([
            InputLayer(input_shape=(WINDOW_SIZE // WINDOW_STEP, 1), name='input'),
            Dense(DENSE_LAYERS, activation='relu'),
            LSTM(LSTM_LAYERS),
            Dense(len(CLASSIFICATION.keys()), activation='softmax', name='output')
        ])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=METRICS)

        self.model.summary()

    def train_model(self):
        print("Beginning training...")
        X = self.train[0]
        Y = self.train[1]
        vX = self.validate[0]
        vY = self.validate[1]

        print("Filtering training signals...")
        start = time.time()
        for x in range(X.shape[0]):
            temp = butter_filter(X[x].flatten(), LOWCUT, HIGHCUT, ORDER)
            X[x] = temp.reshape(-1, WINDOW_SIZE, 1)
        stop = time.time()
        print(f"Filtering took {stop - start} seconds ({(stop - start) / X.shape[0]} per window)")

        print("Filtering validation signals...")
        start = time.time()
        for x in range(vX.shape[0]):
            temp = butter_filter(vX[x].flatten(), LOWCUT, HIGHCUT, ORDER)
            vX[x] = temp.reshape(-1, WINDOW_SIZE, 1)
        stop = time.time()
        print(f"Filtering took {stop - start} seconds ({(stop - start) / vX.shape[0]} per window)")

        filename = 'history.csv'
        # history_logger = tf.keras.callbacks.CSVLogger(filename, separator=',', append=True)
        # out_batch = NBatchLogger(display=BATCH_LOG_SIZE)
        history = self.model.fit(X, Y, validation_data=[vX, vY],
                                 epochs=EPOCHS, verbose=VERBOSE,
                                 batch_size=BATCH_SIZE, shuffle=True)
        # callbacks=[out_batch, history_logger])

        self.model.save(f'LSTM_D{DENSE_LAYERS}_L{LSTM_LAYERS}_STEP{WINDOW_STEP}_FullClassification.h5')

        print(f"\nLoss: {history.history['loss']}")
        print(f"Val loss: {history.history['val_loss']}\n")
        xbatch = np.arange(0, EPOCHS, EPOCHS / len(out_batch.batch_losses))
        plt.plot(xbatch, out_batch.batch_losses, '-b', label='Batch Training loss')
        plt.plot(xbatch, out_batch.batch_accuracy, 'r', label='Batch Training Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.title('Loss and Accuracy, training LSTM network')
        plt.show()


def reverse_onceHot(n):
    temp = np.argmax(n, axis=-1)
    return list(CLASSIFICATION.keys())[temp]


def reverse_classification(n):
    return list(CLASSIFICATION.keys())[n]


def evaluate_model(model_name, test_data):
    print(f'Evaluating model {model_name}')
    start = time.time()
    # open supplied filename as model
    model = tf.keras.models.load_model(model_name)

    # filter each window properly, then append
    x = []
    b, a = butter(ORDER, [LOWCUT / (FS / 2), HIGHCUT / (FS / 2)], "band")
    for window in test_data[0]:
        # filter window, then slice into proper size
        filtered = filtfilt(b, a, window.flatten())
        x.append(filtered[::WINDOW_STEP])
    x = np.array(x).reshape(-1, WINDOW_SIZE // WINDOW_STEP, 1)
    y = test_data[1]
    print(f'x:{x.shape}, y:{y.shape}')
    evaluation_history = model.evaluate(x, y, verbose=VERBOSE)
    stop = time.time()
    print(f'Eval complete in {stop - start} seconds')
    print(f'Evaluation: {evaluation_history}')


def predict_model(model_name, test_data, test_labels):
    sum = 0  # total time elapsed
    count = 0  # count how many signals processed
    y_true = []  # the true classification of each signal
    y_pred = []  # model prediction classification of each signal
    b, a = butter(ORDER, [LOWCUT / (FS / 2), HIGHCUT / (FS / 2)], "band")

    signals = test_data
    labels = test_labels

    model = tf.keras.models.load_model(model_name)

    print(f"Beginning predictions...")

    # loop over every x input
    for x in range(signals.shape[0]):
        count += 1
        start = time.time()

        window = filtfilt(b, a, signals[x].flatten())  # filter the signal
        if has_label:
            y_true.append(labels[x])  # record the true label

        temp = np.argmax(model.predict(window.reshape(1, WINDOW_SIZE)))
        y_pred.append(temp)

        stop = time.time()
        sum += stop - start
        if x % 1000 == 0:
            print(f"Predicted {x}/{signals.shape[0]} classifications in {sum} seconds")
            print(f"Estimated time remaining: {(((sum / count) * signals.shape[0]) - sum) / 60} minutes")

    print(f'Average run time for each record: {sum / count}')

    # if has_label:
    #     y_true = np.array(list(map(reverse_onceHot, y_true)))

    # print(list(CLASSIFICATION.keys()))

    if has_label:
        print(classification_report(y_true, y_pred, zero_division=0))

    results = {}
    for val in y_pred:
        if val not in results:
            results[val] = 1
        else:
            results[val] += 1

    print(results)


def create_lite_model(path):
    # create data generator for representative data
    def representative_data_gen():
        # read testing data into a dataframe
        df = pd.read_csv(os.path.join(os.getcwd(), '../Datasets/testing_data_D128x3-stratified.csv'))
        # take every 10th row, ignore the first columns because that's the once-hot data for the label
        data = np.array(df.iloc[::1, 1:], dtype=np.float32)
        print(data.shape)
        print(data[0].shape)

        # yield every 100th data point, should result in ~540 points
        for i in range(0, data.shape[0], 50):
            yield [data[i]]

    representative_data_gen()

    model = tf.keras.models.load_model(path)

    # create filename for saving lite model
    filename = os.path.splitext(os.path.basename(path))[0]
    lite_path = os.path.join(os.getcwd(), filename) + 'quantized' + '.tflite'

    print(f'Converting model {filename} to TF lite model')
    # create lite model if available
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    print(f'Saving model to {lite_path}')

    with open(lite_path, 'wb') as f:
        f.write(tflite_model)


def process_collab_data():
    # Read csv into signalframe, extract signal
    df = pd.read_csv('../Datasets/Kemal360hz.csv')
    signal = df['Signal'].tolist()

    # scale all signal values to be between 0 and 1
    MAX_SIGNAL_VALUE = max(signal)
    signal = list(map(lambda s: s / MAX_SIGNAL_VALUE, signal))

    # create peak detection object, initialize sampling frequency
    detectors = ecgdetectors.Detectors(FS)
    r_peaks = detectors.engzee_detector(signal)

    # get signal value of each peak in peak detection and create coordinates
    annotation_signal = []
    for x in range(len(r_peaks)):
        annotation_signal.append(signal[r_peaks[x]])
    annotation_coords = [[x, y] for x, y in zip(r_peaks, annotation_signal)]

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

    return np.array(windows).reshape(-1, WINDOW_SIZE, 1)


def saved_model_convert(h5_path):
    model = tf.keras.models.load_model(h5_path)
    print(model.summary())
    # tf.saved_model.save(model, 'D45_L23_SavedModel')


def main():
    data = Data()

    data.train_model()
    # evaluate_model('LSTM_D45_L23_STEP1_Stratified.h5', data.test)
    # predict_model("LSTM_D45_L23_STEP1_Stratified.h5", data.test)
    # predict_model("LSTM_D45_L23_STEP1_Stratified.h5", process_collab_data(), has_label=False, lite=False)
    # saved_model_convert("LSTM_D45_L23_STEP1_Stratified.h5")
    # create_lite_model(os.path.join(os.getcwd(),"LSTM_D45_L23_STEP1_Stratified.h5"), data.train)
    # create_lite_model('LSTM_D90_L45_stratified.h5')
    # create_lite_model('DNN_D128x3_BinaryClassifcation.h5')
    # saved_model_convert('LSTM_D90_L45_stratified.h5')
    # data, labels = load_binary_testing_data('testing_data_D128x3-stratified.csv')
    # predict_model('LSTM_D128x3_BinaryClassification.h5', data, labels)

    # pd.DataFrame(data.test[0].reshape(-1, WINDOW_SIZE)).iloc[::50, :].to_csv('representative_data.csv', index=False)

    # print(f'label: {data.test[1].shape}, data: {data.test[0].shape}')
    # test_data = np.concatenate((data.test[1], data.test[0].reshape(-1, WINDOW_SIZE)), axis=1)
    # print(test_data.shape)
    # pd.DataFrame(test_data).to_csv('testing_data_newModel.csv', index=False)


main()
