import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
from scipy.signal import butter, filtfilt
import time
from sklearn.metrics import classification_report
from CollabData import DataProcessor
import os

WINDOW_SIZE = 216
HIGHCUT = 40  # allow frequencies below this rate (filters 50/60hz main noise)
LOWCUT = 0.5  # allow frequencies above this rate (filters 0.05-0.5 hz muscle noise due to harmonics)
ORDER = 5  # order of butterworth filter
DATA_SAMPLE_RATE = 360
FS = 360

CLASSIFICATION_FULL = {'N': 0, 'L': 1, 'R': 2, 'B': 3, 'A': 4, 'a': 5, 'J': 6, 'S': 7, 'V': 8,
                       'r': 9, 'F': 10, 'e': 11, 'j': 12, 'n': 13, 'E': 14, '/': 15, 'f': 16, 'Z': 17}
CLASSIFICATION_AHA = {'N': 0, 'V': 1, 'F': 2, 'E': 3, 'Q': 4, 'O': 5, 'P': 6, 'Z': 7}

CLASSIFICATION = CLASSIFICATION_FULL


def annotation_full_to_aha_map(label):
    if label in ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'e', 'j', '.']:
        return 'N'
    elif label == 'V':
        return 'V'
    elif label in ['F', 'f']:
        return 'F'
    elif label in ['!', 'p']:
        return 'O'
    elif label == 'E':
        return 'E'
    elif label == 'Q':
        return 'Q'
    elif label in ['/', 'P']:
        return 'P'
    else:
        return 'Z'


def reverse_onceHot(n):
    temp = np.argmax(n, axis=-1)
    return list(CLASSIFICATION.keys())[temp]


def predict_model_lite(model_path, test_data):
    sum = 0  # total time elapsed
    count = 0  # count how many signals processed
    y_true = []  # the true classification of each signal
    y_pred = []  # model prediction classification of each signal
    b, a = butter(ORDER, [LOWCUT / (FS / 2), HIGHCUT / (FS / 2)], "band")

    signals = test_data[0]
    labels = test_data[1]

    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # loop over every x input
    for x in range(signals.shape[0]):
        count += 1
        start = time.time()

        window = filtfilt(b, a, signals[x].flatten())  # filter the signal
        y_true.append(labels[x])  # record the true label

        # set input type and invoke the interpreter
        interpreter.set_tensor(input_details[0]['index'], window.reshape(-1, WINDOW_SIZE, 1).astype('float32'))
        interpreter.invoke()

        # get prediction and convert to classification letter
        temp = np.argmax(interpreter.get_tensor(output_details[0]['index']))
        temp = list(CLASSIFICATION.keys())[temp]
        y_pred.append(temp)

        stop = time.time()
        sum += stop - start
        if x % 1000 == 0:
            print(f"Predicted {x}/{signals.shape[0]} classifications in {sum:.2f} seconds")
            print(f"Estimated time remaining: {((((sum / count) * signals.shape[0]) - sum) / 60):.2f} minutes")

    print(f'Average run time for each record: {(sum / count):.4f}')

    y_true = np.array(list(map(reverse_onceHot, y_true)))

    print(list(CLASSIFICATION.keys()))

    y_true_map = {}
    for label in y_true:
        if label not in y_true_map.keys():
            y_true_map[label] = 1
        else:
            y_true_map[label] = y_true_map[label] + 1

    y_pred_map = {}
    for label in y_pred:
        if label not in y_pred_map.keys():
            y_pred_map[label] = 1
        else:
            y_pred_map[label] = y_pred_map[label] + 1

    print(f'True labels: {y_true_map}\n Predicted Labels: {y_pred_map}')

    print(classification_report(y_true, y_pred, zero_division=0))


def load_labeled_testing_data(model_path, data_path):
    # read testing data into a dataframe
    df = pd.read_csv(os.path.join(os.getcwd(), data_path))

    # take every 10th row, ignore the first columns because that's the once-hot data for the label
    data = np.array(df.iloc[::10, len(CLASSIFICATION):], dtype=np.float32)
    data = data.reshape(data.shape[0], data.shape[1], 1)

    # take every 10th row, the first columns are the once hot for the label
    label = np.array(df.iloc[::10, : len(CLASSIFICATION)])
    print(data.shape, label.shape)

    predict_model_lite(model_path, (data, label))


def load_unlabeled_collab_data(freq, model_path, data_path):
    start = time.time()
    processor = DataProcessor(WINDOW_SIZE, HIGHCUT, LOWCUT, ORDER, FS)
    signal, annotation_coords = processor.preprocess(freq, data_path)
    signal = np.array(processor.create_windows(signal, annotation_coords))
    print(signal.shape)
    labels = ['N' for i in range(signal.shape[0])]
    elapsed = time.time() - start
    print(f'Preprocessing took {elapsed} seconds, or {elapsed / signal.shape[0]} seconds per record')

    predict_model_lite(model_path, (signal, labels))


load_unlabeled_collab_data(360, 'LSTM_D90_L45_stratified.tflite', 'Kemal360hz.csv')
load_unlabeled_collab_data(500, 'LSTM_D90_L45_stratified.tflite', 'Kemal500hz.csv')
load_unlabeled_collab_data(1300, 'LSTM_D90_L45_stratified.tflite', 'Kemal1300hz.csv')

load_labeled_testing_data('LSTM_D90_L45_stratified.tflite', 'testing_data_D90-L45-stratified.csv')