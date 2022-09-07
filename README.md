# Supervised ADS-B anomaly detection

"src/main.py" is a python script used to train and test an LSTM model or a Transformer model to
detect anomalies in windows of adsb_messages in a supervised fashion. This script can also be used
to test pretrained models to detect anomalies as well as a cumulative alarm mechanism. The alarm
mechanism is launched when the number of detected anomalous meta messages exceeds a specified
threshold for a given flight.The anomalies already tested for detection are the following: waypoints
attacks as well as gradual attacks individually inflicted on altitude, ground speed, track, latitude
and longitude.

## NEEDED PACKAGES

pandas,<br>
tqdm,<br>
sklearn,<br>
torch,<br>
pyaml.<br>
These packages could be installed using pip or conda.

## STRUCTURE OF THE PROJECT

```
.
├── configs
├── datasets
├── models
├── README.txt
└── src
    ├── analyze_detection.py
    ├── exceptions.py
    ├── main.py
    ├── models_classes.py
    └── training_testing_models.py
```	
	
configs: Contains configuration files used for training, testing or analyzing detection.<br> 
datasets: Should contain datasets used to train or test a deep learning model to detect different kinds of attacks (mentioned above). 
Such datasets can be downloaded by accessing [https://mega.nz/folder/xTcRFIxZ#hTj6vVoR-40ZgAx_9BSHmw](https://mega.nz/folder/xTcRFIxZ#hTj6vVoR-40ZgAx_9BSHmw).<br> 
models: Contains your pretrained models (as well as newly trained models).<br>
src: Contains the source codes.<br>
analyze\_detection.py: Module containing functions and classes used to analyze the detection of anomalies in individual flights as well the alarm mechanism.<br>
exceptions.py: Module containing the exceptions garanteeing the use of a config file with valid values.<br>
main.py: The only script that should be run.<br>
models\_classes.py: Module containing the classes of the models.<br>
training\_testing\_models.py: Module containing functions used to train and test models.<br>

## DATASETS

The directory "datasets" contains datasets used to train or test a deep learning model to detect
different kinds of attacks. Each dataset contains a list of files associated with specific
flights identified by their callsigns. Each row in such files is called a meta message and
contains the following features: rank/counter, relative time, aircraft id, altitude, ground
speed, track, latitude, longitude, vertical rate, label (attacked:1, normal:0). Note that in any
given dataset only half the flights are attacked in order to have a balanced amount of attacked
messages relative to normal messages. These features are denoted respectively by
0,7,8,4,12,13,14,15,16,17,24 which denote the corresponding cell number in a SBS message (except
for 0 and 24 which do not exist in a SBS message).  For a more detailed explanation of the SBS
format please refer to: [http://woodair.net/SBS/Article/Barebones42_Socket_Data.htm](http://woodair.net/SBS/Article/Barebones42_Socket_Data.htm).

## CONFIG FILE STRUCTURE

### EXAMPLE OF A CONFIG FILE

train\_directory: 'Train\_3waypoints\_att\_mean\_Meta'<br>
test\_directory: 'Test\_3waypoints\_att\_mean\_Meta'<br>
lookback: 10<br>
epochs: 10<br>
batch\_size: 512<br>
to\_diff\_cond: 1<br>
model\_class: 'MV\_LSTM\_SUPERVISED'<br>
pretrained: 1<br> 
label\_for\_metamessage\_or\_window: 0<br>
concatenated\_df: 0<br>
threshold\_alarm: 100<br>
loss\_function: "mse"<br> 
optimizer: "adam"<br> 
future\_time\_as\_input: 0<br>
scaling: 0<br> 

### EXPLANATION OF EACH KEY

- train/test\_directory: Path of the train/test directory relative to the datasets directory.
- lookback: Number of meta messages in input windows to be classified as normal or anomalous.	
- epochs: Number of training epochs to be used.
- bach\_size: Batch size to be used.
- to\_diff\_cond: If 1 is chosen then apply difference to meta messages i.e. DataFrame.diff(periods=1, axis=0) and consider
these rows as meta-messages, in other words windows of these rows are used for anomaly detection. Otherwise if 0 is chosen then
windows of meta-messages (not difference of meta-messages) are used for anomaly detection.
- model\_class: If 'MV\_LSTM\_SUPERVISED' or 'TRANSFORMER\_SUPERVISED' are chosen then an LSTM or a TRANSFORMER are
chosen respectively for anomaly detection.
- pretrained: If 0 is chosen then train a new model otherwise if 1 is chosen then use a pretrained model.
- label\_for\_metamessage\_or\_window: If 0 is chosen then the meta message after the input window is classified as normal or anomalous
otherwise if 1 is chosen then the window is classified as normal or anomalous.
- concatenated\_df: If 1 is chosen then the dataframes of all the flights are concatenated into one big dataframe which cause
the presence of some input windows containing meta messages from two flights in addition to the usual windows containing meta messages
from only one flight. Otherwise if 0 is chosen then dataloaders are created in a way preventing the windows from  containing meta messages from more than one flight.
- threshold\_alarm: The alarm mechanism is launched when the number of detected anomalous meta messages exceeds this threshold for a given flight.
- loss\_function: It can be "mse" or "mae".
- optimizer: It can be "sgd" or "adam".
- future\_time\_as\_input: If 1 is chosen then add to the input the time stamp of the meta message located right after the input window. In other words
the input contains a window of meta messages as well as the time stamp of the meta message right after the window. Otherwise if 1 is chosen then
the input contains only the input window of meta messages.
- scaling: If 0 is chosen no scaling is applied. Otherwise if 1 is chosen then scaling is applied. Note that scaling deteriorates supervised anomaly detection and for this reason it should be chosen as 0.

## USAGE

usage: main.py [-h] --config CONFIG (--train\_test | --analyze\_detection)<br> 

This python script is used to train and test an LSTM model or a Transformer model to detect
anomalies in windows of adsb\_messages in a supervised fashion.  This script can also be used with
test pretrained models to detect anomalies as well as a cumulative alarm mechanism. The alarm
mechanism is launched when the number of detected anomalous meta messages exceeds a specified
threshold for a given flight.The anomalies already tested for detection are the following:
waypoints attacks as well as gradual attacks individually inflicted on altitude, ground speed,
track, latitude and longitude.

optional arguments:<br> 
  -h, --help   => show this help message and exit<br> 
  --config CONFIG => specify the config file's path relative to the configs  directory<br> 
  --train\_test => train and/or test a model<br> 
  --analyze\_detection => test a pretrained model as well as the alarm mechanism<br> 

## EXAMPLES

example of training and/or evaluating a new model/pretrained model using the example config file:<br> 
```
python main.py --config config_lstm_3waypoints_att_mean_Meta.yaml --train_test
```

Result:<br> 
```
Model used:

MV_LSTM_SUPERVISED( 
  (linear_combine): Linear(in_features=71, out_features=70, bias=True)
  (l_lstm): LSTM(7, 64, bias=False, batch_first=True) 
  (l_lstm2): LSTM(64, 32, batch_first=True) 
  (l_linear_supervised): Linear(in_features=320, out_features=1, bias=True) 
  (relu): ReLU() 
)

Evaluation of classification of meta messages:

test_mse= 0.02003899857237369 
classifier evaluation 
		 conf_matrix  precision     recall     fscore 
0  [[71434, 887], [1415, 41140]]  97.889452  96.674891  97.278381 
TPR= 96.67489131711903 
FPR= 1.226476403810788 
finished
```

example of evaluating a pretrained model and the alarm which uses its predictions
using the example config file:<br> 
```
python main.py --config config_lstm_3waypoints_att_mean_Meta.yaml --analyze_detection
```
Result:<br> 
```
Model used: 

MV_LSTM_SUPERVISED( 
  (linear_combine): Linear(in_features=71, out_features=70, bias=True) 
  (l_lstm): LSTM(7, 64, bias=False, batch_first=True) 
  (l_lstm2): LSTM(64, 32, batch_first=True) 
  (l_linear_supervised): Linear(in_features=320, out_features=1, bias=True) 
  (relu): ReLU() 
) 

Evaluation of classification of meta messages: 

test_mse= 0.02003899857237369 
classifier evaluation 
		 conf_matrix  precision     recall     fscore 
0  [[71434, 887], [1415, 41140]]  97.889452  96.674891  97.278381 
TPR= 96.67489131711903 
FPR= 1.226476403810788 

Performance of alarm: the performance of the classification of individual
flights as containing any attack or not.


Performance of alarm from predictions: 

test_mse= 0.0 
classifier evaluation 
	 conf_matrix  precision  recall  fscore 
0  [[11, 0], [0, 9]]      100.0   100.0   100.0 
TPR= 100.0 
FPR= 0.0 

Theoretical alarm: 
This is an alarm that depends on a perfect theoretical detector that can detect
all anomalous meta messages. In other words this alarm is launched if the number
of anomalies in a given flight is above the specified alarm threshold.


Performance of theoretical alarm: 

test_mse= 0.0 
classifier evaluation 
	 conf_matrix  precision  recall  fscore 
0  [[11, 0], [0, 9]]      100.0   100.0   100.0 
TPR= 100.0 
FPR= 0.0 

Performance of alarm from predictions compared to the performance of theoretical alarm: 

test_mse= 0.0 
classifier evaluation 
	 conf_matrix  precision  recall  fscore 
0  [[11, 0], [0, 9]]      100.0   100.0   100.0 
TPR= 100.0 
FPR= 0.0 
finished
```
