import argparse
import yaml
import glob
import exceptions

parser = argparse.ArgumentParser(
    description="This python script is used to train and test an LSTM model or a Transformer model to detect anomalies in windows of adsb_messages in a supervised fashion. This script can also be used to test pretrained models to detect anomalies as well as a cumulative alarm mechanism. The alarm mechanism is launched  when the number of detected anomalous meta messages exceeds a specified threshold for a given flight.The anomalies already tested for detection are the following: waypoints attacks as well as gradual attacks individually inflicted on altitude, ground speed, track, latitude and longitude."
)
parser.add_argument('--config',
                    help="specify the config file\'s path relative to the configs directory",
                    required=True,
                    type=str)
group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument("--train_test",help="train and/or test a model", action="store_true")
group1.add_argument("--analyze_detection",help="test a pretrained model as well as the alarm mechanism" ,action="store_true")
args = parser.parse_args()

#load the chosen config file
config_relative_path=args.config
with open(f"../configs/{config_relative_path}","r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#Verify the validity of the config
required_config_keys = {
    'loss_function', 'future_time_as_input', 'scaling', 'epochs',
    'to_diff_cond', 'label_for_metamessage_or_window', 'optimizer',
    'test_directory', 'model_class', 'threshold_alarm', 'concatenated_df',
    'lookback', 'pretrained', 'train_directory', 'batch_size'
}
if set(config) != required_config_keys:
    raise exceptions.WrongConfigKeysError(required_config_keys, set(config))
files = glob.glob(f"../datasets/{config['train_directory']}/*.csv")
if not files:
    raise exceptions.DatasetNotFoundError(train_directory)
files = glob.glob(f"../datasets/{config['test_directory']}/*.csv")
if not files:
    raise exceptions.DatasetNotFoundError(test_directory)
exceptions.verify_strictly_positive(config["lookback"], str="lookback")
exceptions.verify_strictly_positive(config["epochs"], str="epochs")
exceptions.verify_strictly_positive(config["batch_size"], str="batch_size")
exceptions.verify_if_in_list(config['to_diff_cond'],
                             str="to_diff_cond",
                             list=[0, 1])
exceptions.verify_if_in_list(
    config['model_class'],
    str="",
    list=['TRANSFORMER_SUPERVISED', 'MV_LSTM_SUPERVISED'])
exceptions.verify_if_in_list(config['pretrained'],
                             str="pretrained",
                             list=[0, 1])
exceptions.verify_if_in_list(config['label_for_metamessage_or_window'],
                             str="label_for_metamessage_or_window",
                             list=[0, 1])
exceptions.verify_if_in_list(config['concatenated_df'],
                             str="concatenated_df",
                             list=[0, 1])
exceptions.verify_strictly_positive(config["threshold_alarm"],
                                    str="threshold_alarm")
exceptions.verify_if_in_list(config['loss_function'],
                             str="loss_function",
                             list=["mse", "mae"])
exceptions.verify_if_in_list(config['optimizer'],
                             str="optimizer",
                             list=["adam", "sgd"])
exceptions.verify_if_in_list(config['future_time_as_input'],
                             str="future_time_as_input",
                             list=[0, 1])
exceptions.verify_if_in_list(config['scaling'], str="scaling", list=[0, 1])

#dump the loaded chosen config file as a hidden file that is needed for other modules
with open('.config.yaml', 'w') as outfile:
    try:
        yaml.dump(config, outfile)
    except yaml.YAMLError as exc:
        print(exc)

#The two following modules should be loaded after the dumping of '.config.yaml' since they depend on it
if args.train_test:
    import training_testing_models
    training_testing_models.main_training_testing_models()
elif args.analyze_detection:
    import analyze_detection
    analyze_detection.main_analyze_test_flights()
