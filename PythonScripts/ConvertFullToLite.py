import tensorflow as tf
import pandas as pd
import os

WINDOW_SIZE = 216  # Width of each data sample


def create_lite_model(path, data):
    # create data generator for representative data
    def representative_data_gen():
        for i in data:
            yield i.reshape(1, WINDOW_SIZE, 1)

    model = tf.keras.models.load_model(path)
    model.summary()

    # create filename for saving lite model
    filename = os.path.splitext(os.path.basename(path))[0]
    lite_path = os.path.join(os.getcwd(), filename) + '.tflite'

    print(f'Converting model {filename} to TF lite model')
    # create lite model if available
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    print(f'Saving model to {lite_path}')

    with open(lite_path, 'wb') as f:
        f.write(tflite_model)


# read data from the csv, reshape into proper input shape
df = pd.read_csv('../Datasets/representative_data.csv')
data = df.to_numpy(dtype='float32').reshape(-1, WINDOW_SIZE, 1)

print(f'The shape of the data is {data.shape}')
create_lite_model(os.path.join(os.getcwd(), '../LSTM_D45_L23_STEP1_Stratified.h5'), data)
