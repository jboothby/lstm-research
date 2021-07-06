import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
from scipy.signal import butter, filtfilt
import time
from sklearn.metrics import classification_report
from CollabData import normalize, preprocess, create_windows, graph_signal
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


if WINDOW_SIZE == 360:
    CLASSIFICATION = CLASSIFICATION_FULL
else:
    CLASSIFICATION = CLASSIFICATION_AHA


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


def reverse_classification(n):
    return list(CLASSIFICATION.keys())[n]


def predict_model_lite(model_path, test_data):
    sum = 0  # total time elapsed
    count = 0  # count how many signals processed
    y_true = []  # the true classification of each signal
    y_pred = []  # model prediction classification of each signal
    b, a = butter(ORDER, [LOWCUT / (FS / 2), HIGHCUT / (FS / 2)], "band")

    signals = test_data[0]
    labels = test_data[1]

    # model = tf.keras.models.load_model(model_name)
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

    print(classification_report(y_true, y_pred, zero_division=0))


def load_labeled_testing_data():
    # read testing data into a dataframe
    df = pd.read_csv(os.path.join(os.getcwd(), 'testing_data.csv'))

    # take every 10th row, ignore the first columns because that's the once-hot data for the label
    data = np.array(df.iloc[::10, len(CLASSIFICATION_AHA):], dtype=np.float32)
    data = data.reshape(data.shape[0], data.shape[1], 1)

    # take every 10th row, the first columns are the once hot for the label
    label = np.array(df.iloc[::10, : len(CLASSIFICATION_AHA)])
    print(data.shape, label.shape)

    predict_model_lite('LSTM_D45_L23_STEP1_Stratified.tflite', (data, label))

def load_unlabeled_collab_data(freq, filename):
    start = time.time()
    signal, annotation_coords = preprocess(freq, filename)
    signal = np.array(create_windows(signal, annotation_coords))
    print(signal.shape)
    labels = ['N' for i in range(signal.shape[0])]
    elapsed = time.time() - start
    print(f'Preprocessing took {elapsed} seconds, or {elapsed/signal.shape[0]} seconds per record')

    predict_model_lite('LSTM_D45_L23_STEP1_Stratified.tflite', (signal, labels))


load_unlabeled_collab_data(360, 'Kemal360hz.csv')
load_unlabeled_collab_data(500, 'Kemal500hz.csv')
load_unlabeled_collab_data(1300, 'Kemal1300hz.csv')