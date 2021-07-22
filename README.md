#LSTM Research

The purpose of this project is to develop machine learning models for the classification of ECG heartbeat data into
arrythmia categories. 

##ModelCreation.py
This is the heart of the project. This file reads the training data from the mit_database folder, preprocesses it,
and trains the model. It creates full Tensorflow models in the .h5 format by default, but can convert to .tflite.
It can also quantize a tflite model to 8 bit integer activation. Lastly, it creates the datasets by writing
the processed data windows to .csv files for use in benchmarking.

##DataProcessor.py
This file describes a class called DataProcessor that will read the unlabelled data from the .csv documents given to us
by our collaborators and process it. It takes the data and scales it to be float values between 0 and 1, then runs a peak
detection algorithm to find the heartbeats, resamples the data to 360hz, then splits this into 600ms windows with 216
points each.

##ConvertFullToLite.py
This is used by ModelCreation to convert the .h5 models into lite/quantized models.

##Benchmark.py
This file runs automated benchmarking of models against existing datasets to test for speed and accuracy. It is meant to
be run from the command line on multiple machines for comparison purposes.

###Usage instructions:
- Clone this repository
- Ensure the following packages are installed 
  - pip3 install numpy
  - pip3 install pandas
  - pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
  - pip3 install scipy
  - pip3 install sklearn
  - pip3 install py-ecg-detectors
  - pip3 install matplotlib
  
- Invoke from the command line as follows:
  python3 Benchmark.py --computerType *[identifyingName]* --quantizedOnly *[True/False]*
  
The program will execute and run multiple model types against the datasets in ./Datasets
If you set `--quantizedOnly` to `True`, it will only run the int8 quantized models which is necessary for Google TPU.

A summary of the performance of each model will be saved to a file in ./Reports called:

"*summary-`--computerType`-`ModelName`-`TodaysDate(yyyy-mm--dd)`.json*"