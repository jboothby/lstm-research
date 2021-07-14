import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
from scipy.signal import butter, filtfilt
import time
from sklearn.metrics import classification_report
from CollabData import DataProcessor
import os
import json

WINDOW_SIZE = 216
HIGHCUT = 40  # allow frequencies below this rate (filters 50/60hz main noise)
LOWCUT = 0.5  # allow frequencies above this rate (filters 0.05-0.5 hz muscle noise due to harmonics)
ORDER = 5  # order of butterworth filter
DATA_SAMPLE_RATE = 360
FS = 360

CLASSIFICATION = {'N': 0, 'L': 1, 'R': 2, 'B': 3, 'A': 4, 'a': 5, 'J': 6, 'S': 7, 'V': 8,
                  'r': 9, 'F': 10, 'e': 11, 'j': 12, 'n': 13, 'E': 14, '/': 15, 'f': 16, 'Z': 17}


def reverse_onceHot(n):
    temp = np.argmax(n, axis=-1)
    return list(CLASSIFICATION.keys())[temp]


class Benchmark:
    y_true = []
    y_pred = []
    b, a = butter(ORDER, [LOWCUT / (FS / 2), HIGHCUT / (FS / 2)], "band")
    summary = {}

    def __init__(self, model_path, data_path, binary_classification=False, quantized=False):
        self.model_path = model_path
        self.data_path = data_path
        self.binary_classification = binary_classification
        self.quantized = quantized
        self.sum = 0
        self.count = 0

        self.data_type = 'uint8' if quantized else 'float32'
        self.window_shape = (-1, WINDOW_SIZE) if binary_classification else (-1, WINDOW_SIZE, 1)
        self.inference_to_pred = \
            self.binaryclass_inference_to_pred if binary_classification \
                else self.multiclass_inference_to_pred

        if binary_classification:
            self.signals, self.labels = self.load_binary_testing_data()

    @staticmethod
    def multiclass_inference_to_pred(x):
        temp = np.argmax(x)
        return list(CLASSIFICATION.keys())[temp]

    @staticmethod
    def binaryclass_inference_to_pred(x):
        return round(x.flatten()[0])

    def begin_inference_lite(self):

        interpreter = tflite.Interpreter(self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]['index']
        output_details = interpreter.get_output_details()[0]['index']

        for x in range(self.signals.shape[0]):
            self.count += 1
            start = time.time()

            window = filtfilt(self.b, self.a, self.signals[x].flatten())
            interpreter.set_tensor(input_details, window.reshape(self.window_shape).astype(self.data_type))
            interpreter.invoke()

            self.y_true.append(self.labels[x].flatten()[0])
            self.y_pred.append(self.inference_to_pred(interpreter.get_tensor(output_details)))

            stop = time.time()
            self.sum += stop - start

            if x % 1000 == 0:
                print(f"Predicted {x}/{self.signals.shape[0]} classifications in {self.sum:.2f} seconds")
                print(f"Estimated time remaining: \
                 {((((self.sum / self.count) * self.signals.shape[0]) - self.sum) / 60):.2f} minutes")

        if not self.binary_classification:
            self.y_true = list(map(reverse_onceHot, self.y_true))

    def summarize(self):
        avg_run_time = round(self.sum / self.count, 6)
        print(f'Average run time for each record: {avg_run_time}')

        results = {}
        for t, p in zip(self.y_true, self.y_pred):
            if t not in results.keys():
                results[t] = {p: 1}
            if p not in results[t].keys():
                results[t][p] = 1
            else:
                results[t][p] += 1

        self.summary['data'] = results

        sum = 0
        for key, val in results.items():
            for v in val.values():
                sum += v
            print(f'True: {key}, Total: {sum}, Predicted: {val}')

        self.summary['total-values'] = self.sum
        self.summary['time-per-infer'] = avg_run_time
        self.summary['accuracy'] = \
            classification_report(self.y_true, self.y_pred, zero_division=0, output_dict=True)['accuracy']

        print(classification_report(self.y_true, self.y_pred, zero_division=0))

        return self.summary

    # def load_labeled_testing_data(self):
    #     # read testing data into a dataframe
    #     df = pd.read_csv(os.path.join(os.getcwd(), data_path))
    #
    #     # take every 10th row, ignore the first columns because that's the once-hot data for the label
    #     data = np.array(df.iloc[::100, len(CLASSIFICATION):], dtype=np.float32)
    #     data = data.reshape(data.shape[0], data.shape[1], 1)
    #
    #     # take every 10th row, the first columns are the once hot for the label
    #     label = np.array(df.iloc[::100, : len(CLASSIFICATION)])
    #
    #     if convert_to_binary:
    #         label = np.array(list(map(reverse_onceHot, label)))
    #         label = np.array(list(map(lambda x: 1 if x == 'N' else 0, label)))
    #
    #     print(data.shape, label.shape)
    #
    #     return data, label
    #
    # # return predict_model_lite(model_path, (data, label))
    #
    # def load_unlabeled_collab_data(self):
    #     start = time.time()
    #     processor = DataProcessor(WINDOW_SIZE, HIGHCUT, LOWCUT, ORDER, FS)
    #     signal, annotation_coords = processor.preprocess(freq, data_path)
    #     signal = np.array(processor.create_windows(signal, annotation_coords))
    #     print(signal.shape)
    #     labels = ['N' for i in range(signal.shape[0])]
    #     elapsed = time.time() - start
    #     time_per_preprocess = round(elapsed / signal.shape[0], 5)
    #     print(f'Preprocessing took {elapsed} seconds, or {time_per_preprocess} seconds per record')
    #
    #     summary = predict_model_lite(model_path, (signal, labels))
    #     summary['time-per-preprocess'] = time_per_preprocess
    #     return summary

    def load_binary_testing_data(self):
        # read testing data into a dataframe
        df = pd.read_csv(os.path.join(os.getcwd(), self.data_path))

        data = np.array(df.iloc[::1, 1:], dtype=np.float32)
        data = data.reshape(data.shape[0], data.shape[1])

        label = np.array(df.iloc[::1, : 1])
        print(data.shape, label.shape)
        return data, label


benchmark = Benchmark('LSTM_D128x3_BinaryClassifcation.tflite', 'testing_data_D128x3-stratified.csv',
                      binary_classification=True)
benchmark.begin_inference_lite()
benchmark.summarize()

# TODO: Figure out why this part is not returning 0<= value <= 1
# benchmark = Benchmark('LSTM_D128x3_BinaryClassifcationquantized.tflite', 'testing_data_D128x3-stratified.csv',
#                       binary_classification=True, quantized=True)
# benchmark.begin_inference_lite()
# benchmark.summarize()

# data, label = load_labeled_testing_data('testing_data_D90-L45-stratified.csv')
# benchmark = Benchmark('LSTM_D90_L45_stratified.tflite', data, label)
# benchmark.begin_inference()

# summary = {}
# summary['Kemal360hz'] = load_unlabeled_collab_data(360, 'LSTM_D90_L45_stratified.tflite', 'Kemal360hz.csv')
# summary['Kemal500hz'] = load_unlabeled_collab_data(500, 'LSTM_D90_L45_stratified.tflite', 'Kemal500hz.csv')
# summary['Kemal1300hz'] = load_unlabeled_collab_data(1300, 'LSTM_D90_L45_stratified.tflite', 'Kemal1300hz.csv')
# summary['Testing Data'] = load_labeled_testing_data('LSTM_D90_L45_stratified.tflite',
#                                                     'testing_data_D90-L45-stratified.csv')
#
# with open('inference-summary.json', 'w') as outfile:
#     json.dump(summary, outfile, indent=2)
