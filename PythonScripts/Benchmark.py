import datetime

import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
from scipy.signal import butter, filtfilt
import time
from sklearn.metrics import classification_report
from DataProcessor import DataProcessor
import os
import json
import argparse

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

    def __init__(self, model_path, uses_binary_classification=False, quantized=False, ):
        self.model_path = model_path
        self.window_shape = (-1, WINDOW_SIZE) if uses_binary_classification else (-1, WINDOW_SIZE, 1)

        self.uses_binary_classification = uses_binary_classification
        self.quantized = quantized

        if uses_binary_classification:
            self.inference_to_pred = self.binaryclass_inference_to_pred
        else:
            self.inference_to_pred = self.multiclass_inference_to_pred

    def set_data_props(self, data_path, freq=None,
                       is_unlabeled_collab_data=False):
        self.sum = 0
        self.count = 0
        self.time_elapsed = 0
        self.preprocessing_time = None
        self.data_path = data_path
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

    @property
    def begin_inference_lite(self):

        interpreter = tflite.Interpreter(self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        data_type = np.float32

        if self.quantized:
            data_type = input_details['dtype']

            in_scale, in_zero_point = input_details['quantization']
            input_float_to_int = lambda x: x / in_scale + in_zero_point

            out_scale, out_zero_point = output_details['quantization']
            output_int_to_float = lambda x: (x - out_zero_point) * out_scale

        quant_elapsed = 0

        for x in range(self.signals.shape[0]):
            self.count += 1
            start = time.time()

            window = filtfilt(self.b, self.a, self.signals[x].flatten())

            if self.quantized:
                quant_start = time.time()
                window = (np.vectorize(input_float_to_int)(window)).astype(data_type)
                quant_elapsed += time.time() - quant_start
            else:
                window = window.astype(data_type)

            interpreter.set_tensor(input_details['index'], window.reshape(self.window_shape))
            interpreter.invoke()

            output = interpreter.get_tensor(output_details['index'])

            if self.quantized:
                quant_start = time.time()
                output = output.astype(np.float32)
                output = np.vectorize(output_int_to_float)(output)
                quant_elapsed += time.time() - quant_start

            self.y_true.append(self.labels[x].flatten()[0])
            self.y_pred.append(self.inference_to_pred(output))

            stop = time.time()
            self.time_elapsed += stop - start

            if x % 1000 == 0:
                print(f"Predicted {x}/{self.signals.shape[0]} classifications in {self.time_elapsed:.2f} seconds")
                print(f"Estimated time remaining: \
                 {((((self.time_elapsed / self.count) * self.signals.shape[0]) - self.time_elapsed) / 60):.2f} minutes")

        if not self.uses_binary_classification:
            self.y_true = list(map(reverse_onceHot, self.y_true))

        print(f'Quantizing the data took {quant_elapsed} seconds out of {self.time_elapsed} total')
        print(f'That\'s an extra {quant_elapsed / self.count} seconds per record')

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
        label = np.array(df.iloc[::1, : 1], dtype=int)

        data = np.array(df.iloc[::1, 1:])
        data = data.reshape(data.shape[0], data.shape[1])

        print(data.shape, label.shape)
        return data, label


def main(computer_name='PC', quantized_only=False):
    datasets = {
        '../Datasets/testing_data_D128x3-stratified.csv': {'name': 'Testing', 'freq': 360, 'collab_data': False},
        '../Datasets/Kemal360hz.csv': {'name': 'Kemal360', 'freq': 360, 'collab_data': True},
        '../Datasets/Kemal500hz.csv': {'name': 'Kemal500', 'freq': 500, 'collab_data': True},
        '../Datasets/Kemal1300hz.csv': {'name': 'Kemal1300', 'freq': 1300, 'collab_data': True}
    }

    models = {
        '../Models/DNN_D128x3_BinaryClassification_Quantized.tflite': {
            'name': 'DNN_D128x3_Quantized', 'binary': True, 'quantized': True},
        '../Models/DNN_D128x3_BinaryClassifcation.tflite': {
            'name': 'DNN_D128x3', 'binary': True, 'quantized': False}
    }

    for model, model_props in models.items():
        if quantized_only and not model_props['quantized']:
            continue

        json_summary_object = {}
        bench = Benchmark(model, model_props['binary'], model_props['quantized'])

        for data, data_props in datasets.items():
            bench.set_data_props(data_path=data, freq=data_props['freq'],
                                 is_unlabeled_collab_data=data_props['collab_data'])
            json_summary_object[data_props['name']] = bench.begin_inference_lite

        today_date = str(datetime.date.today())
        with open(f"../Reports/summary-{model_props['name']}-{computer_name}-{today_date}.json", 'w') as f:
            json.dump(json_summary_object, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Keras models against datasets")
    parser.add_argument('--computerType', type=str, required=True,
                        help='The type of computer the benchmark is running on like PC or Rpi3')
    parser.add_argument('--quantizedOnly', type=bool, default=False,
                        help='Set to True to only run int8 quantized model')

    args = parser.parse_args()
    main(args.computerType, args.quantizedOnly)
