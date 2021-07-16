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
    b, a = butter(ORDER, [LOWCUT / (FS / 2), HIGHCUT / (FS / 2)], "band")

    def __init__(self, model_path, uses_binary_classification=False):
        self.model_path = model_path
        self.window_shape = (-1, WINDOW_SIZE) if uses_binary_classification else (-1, WINDOW_SIZE, 1)

        self.uses_binary_classification = uses_binary_classification

        if uses_binary_classification:
            self.inference_to_pred = self.binaryclass_inference_to_pred
        else:
            self.inference_to_pred = self.multiclass_inference_to_pred

    def set_data_props(self, data_path, quantized=False, freq=None,
                       is_unlabeled_collab_data=False):
        self.sum = 0
        self.count = 0
        self.time_elapsed = 0
        self.preprocessing_time = None

        self.data_path = data_path
        self.data_type = 'uint8' if quantized else 'float32'
        self.signals, self.labels = self.load_testing_data(freq, is_unlabeled_collab_data)

        self.y_pred = []
        self.y_true = []
        self.summary = {}

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

        time_elapsed = 0
        for x in range(self.signals.shape[0]):
            self.count += 1
            start = time.time()

            window = filtfilt(self.b, self.a, self.signals[x].flatten())

            interpreter.set_tensor(input_details, window.reshape(self.window_shape).astype(self.data_type))
            interpreter.invoke()

            self.y_true.append(self.labels[x].flatten()[0])
            self.y_pred.append(self.inference_to_pred(interpreter.get_tensor(output_details)))

            stop = time.time()
            self.time_elapsed += stop - start

            if x % 1000 == 0:
                print(f"Predicted {x}/{self.signals.shape[0]} classifications in {self.time_elapsed:.2f} seconds")
                print(f"Estimated time remaining: \
                 {((((self.time_elapsed / self.count) * self.signals.shape[0]) - self.time_elapsed) / 60):.2f} minutes")


        if not self.uses_binary_classification:
            self.y_true = list(map(reverse_onceHot, self.y_true))

        self.summarize()

        return self.summary

    def summarize(self):
        avg_run_time = round(self.time_elapsed / self.count, 6)
        print(f'Average run time for each record: {avg_run_time}')

        results = {}
        for t, p in zip(self.y_true, self.y_pred):
            t = str(t)
            if t not in results.keys():
                results[t] = {p: 1}
            if p not in results[t].keys():
                results[t][p] = 1
            else:
                results[t][p] += 1

        self.summary['data'] = results

        for key, val in results.items():
            number_of_this_value = 0
            for v in val.values():
                number_of_this_value += v
                self.sum += v
            print(f'True: {key}, Total: {number_of_this_value}, Predicted: {val}')

        self.summary['total-values'] = self.sum
        self.summary['time-per-infer'] = avg_run_time

        if self.preprocessing_time is not None:
            self.summary['time-per-preprocess'] = self.preprocessing_time

        self.summary['accuracy'] = \
            round(classification_report(self.y_true, self.y_pred, zero_division=0, output_dict=True)['accuracy'], 6)

        print(classification_report(self.y_true, self.y_pred, zero_division=0))

    def load_testing_data(self, freq=None, is_unlabeled_collab_data=False):

        assert (not is_unlabeled_collab_data or freq is not None)

        if is_unlabeled_collab_data:
            start = time.time()

            processor = DataProcessor(WINDOW_SIZE, HIGHCUT, LOWCUT, ORDER, FS)
            raw_signal, annotation_coords = processor.preprocess(freq, self.data_path)

            data = np.array(processor.create_windows(raw_signal, annotation_coords))
            label = np.array([1 for i in range(data.shape[0])])

            time_elapsed = time.time() - start
            time_per_preprocess = round(time_elapsed / data.shape[0], 5)

            self.preprocessing_time = time_per_preprocess
            print(f'Preprocessing took {time_elapsed} seconds, or {time_per_preprocess} seconds per record')

            return data, label

        # read testing data into a dataframe
        df = pd.read_csv(os.path.join(os.getcwd(), self.data_path))

        data = np.array(df.iloc[::1, 1:], dtype=np.float32)
        data = data.reshape(data.shape[0], data.shape[1])
        label = np.array(df.iloc[::1, : 1], dtype=int)

        print(data.shape, label.shape)
        return data, label


def main():
    computer_name = "PC"
    json_summary_object = {}

    bench = Benchmark('LSTM_D128x3_BinaryClassifcation.tflite', uses_binary_classification=True)

    bench.set_data_props(data_path='testing_data_D128x3-stratified.csv')
    json_summary_object['Testing'] = bench.begin_inference_lite()

    bench.set_data_props(data_path='Kemal360hz.csv', freq=360, is_unlabeled_collab_data=True)
    json_summary_object['Kemal 360Hz'] = bench.begin_inference_lite()

    bench.set_data_props(data_path='Kemal500hz.csv', freq=500, is_unlabeled_collab_data=True)
    json_summary_object['Kemal 500Hz'] = bench.begin_inference_lite()

    bench.set_data_props(data_path='Kemal1300hz.csv', freq=1300, is_unlabeled_collab_data=True)
    json_summary_object['Kemal 1300'] = bench.begin_inference_lite()

    with open(f'inference-summary-binary-{computer_name}.json', 'w') as f:
        json.dump(json_summary_object, f, indent=2)


if __name__ == "__main__":
    main()
