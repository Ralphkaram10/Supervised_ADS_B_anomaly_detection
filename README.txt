"src/main.py" is a python script used to train and test an LSTM model or a Transformer model to
detect anomalies in windows of adsb_messages in a supervised fashion. This script can also be used
to test pretrained models to detect anomalies as well as a cumulative alarm mechanism. The alarm
mechanism is launched when the number of detected anomalous meta messages exceeds a specified
threshold for a given flight.The anomalies already tested for detection are the following: waypoints
attacks as well as gradual attacks individually inflicted on altitude, ground speed, track, latitude
and longitude.

NEEDED PACKAGES

    pandas,
    tqdm,
    sklearn,
    torch,
    pyaml.
    These packages could be installed using pip or conda.

STRUCTURE OF THE PROJECT

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

    configs: Contains configuration files used for training, testing or analyzing detection.
    datasets: Contains datasets used to train or test a deep learning model to detect different kinds of attacks (mentioned above).
    models: Contains your pretrained models (as well as newly trained models).
    src: Contains the source codes.
    analyze_detection.py: Module containing functions and classes used to analyze the detection of anomalies in individual flights as well the alarm mechanism.
    exceptions.py: Module containing the exceptions garanteeing the use of a config file with valid values.
    main.py: The only script that should be run.
    models_classes.py: Module containing the classes of the models.
    training_testing_models.py: Module containing functions used to train and test models .

DATASETS

    The directory "datasets" contains datasets used to train or test a deep learning model to detect
    different kinds of attacks. Each dataset contains a list of files associated with specific
    flights identified by their callsigns. Each row in such files is called a meta- message and
    contains the following features: rank/counter, relative time, aircraft id, altitude, ground
    speed, track, latitude, longitude, vertical rate, label (attacked:1, normal:0). Note that in any
    given dataset only half the flights are attacked in order to have a balanced amount of attacked
    messages relative to normal messages. These features are denoted respectively by
    0,7,8,4,12,13,14,15,16,17,24 which denote the corresponding cell number in a SBS message (except
    for 0 and 24 which do not exist in a SBS message).  For a more detailed explanation of the SBS
    format please refer to: http://woodair.net/SBS/Article/Barebones42_Socket_Data.htm

CONFIG FILE STRUCTURE

    EXAMPLE OF A CONFIG FILE

    train_directory: 'Train_3waypoints_att_mean_Meta'
    test_directory: 'Test_3waypoints_att_mean_Meta'
    lookback: 10
    epochs: 10
    batch_size: 512
    to_diff_cond: 1
    model_class: 'MV_LSTM_SUPERVISED'
    pretrained: 1 
    label_for_metamessage_or_window: 0
    concatenated_df: 0
    threshold_alarm: 100 
    loss_function: "mse" 
    optimizer: "adam" 
    future_time_as_input: 0
    scaling: 0 

    EXPLANATION OF EACH KEY

    -train/test_directory: Path of the train/test directory relative to the datasets directory.
    -lookback: Number of meta messages in input windows to be classified as normal or anomalous.	
    -epochs: Number of training epochs to be used.
    -bach_size: Batch size to be used.
    -to_diff_cond: If 1 is chosen then apply difference to meta messages i.e. DataFrame.diff(periods=1, axis=0) and consider
    these rows as meta-messages, in other words windows of these rows are used for anomaly detection. Otherwise if 0 is chosen then
    windows of meta-messages (not difference of meta-messages) are used for anomaly detection.
    -model_class: If 'MV_LSTM_SUPERVISED' or 'TRANSFORMER_SUPERVISED' are chosen then an LSTM or a TRANSFORMER are
    chosen respectively for anomaly detection.
    -pretrained: If 0 is chosen then train a new model otherwise if 1 is chosen then use a pretrained model.
    -label_for_metamessage_or_window: If 0 is chosen then the meta message after the input window is classified as normal or anomalous
    otherwise if 1 is chosen then the window is classified as normal or anomalous.
    -concatenated_df: If 1 is chosen then the dataframes of all the flights are concatenated into one big dataframe which cause
    the presence of some input windows containing meta messages from two flights in addition to the usual windows containing meta messages
    from only one flight. Otherwise if 0 is chosen then dataloaders are created in a way preventing the windows from  containing meta messages from
    more than one flight.
    -threshold_alarm: The alarm mechanism is launched when the number of detected anomalous meta messages exceeds this threshold for a given flight.
    -loss_function: It can be "mse" or "mae".
    -optimizer: It can be "sgd" or "adam".
    -future_time_as_input: If 1 is chosen then add to the input the time stamp of the meta message located right after the input window. In other words
    the input contains a window of meta messages as well as the time stamp of the meta message right after the window. Otherwise if 1 is chosen then
    the input contains only the input window of meta messages.
    -scaling: If 0 is chosen no scaling is applied. Otherwise if 1 is chosen then scaling is applied. Note that scaling deteriorates supervised anomaly
    detection and for this reason it should be chosen as 0.

USAGE

    usage: main.py [-h] --config CONFIG (--train_test | --analyze_detection)

    This python script is used to train and test an LSTM model or a Transformer model to detect
    anomalies in windows of adsb_messages in a supervised fashion.  This script can also be used with
    test pretrained models to detect anomalies as well as a cumulative alarm mechanism. The alarm
    mechanism is launched when the number of detected anomalous meta messages exceeds a specified
    threshold for a given flight.The anomalies already tested for detection are the following:
    waypoints attacks as well as gradual attacks individually inflicted on altitude, ground speed,
    track, latitude and longitude.

    optional arguments:
      -h, --help   => show this help message and exit
      --config CONFIG => specify the config file's path relative to the configs  directory
      --train_test => train and/or test a model
      --analyze_detection => test a pretrained model as well as the alarm mechanism

EXAMPLES

    example of training and/or evaluating a new model/pretrained model using the example config file:
    python main.py --config config_lstm_3waypoints_att_mean_Meta.yaml --train_test

    Result:"
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
    finished"

    example of evaluating a pretrained model and the alarm which uses its predictions
    using the example config file:
    python main.py --config config_lstm_3waypoints_att_mean_Meta.yaml --analyze_detection

    Result:"

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
    finished"
