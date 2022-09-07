import torch
import glob
import os
import training_testing_models
from training_testing_models import get_scaler_from_files, create_datasets_dict, get_df_file, to_input_output, get_model, evaluation_detection_metrics, print_evaluation_detection_metrics, get_output_used_for_evaluation, conditional_decorator, predict_dataloader, get_real_targets, flight_is_too_small, remove_small_flights_from_dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
from math import sqrt, floor, ceil
from torch.utils.data import DataLoader, ConcatDataset
from importlib import reload  # Python 3.4+
import scipy

import yaml
import argparse
import models_classes

#load the chosen config file
with open(".config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


@remove_small_flights_from_dict
def create_labels_dict(
        files,
        scaler=None,
        columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
        device=0):
    """
    Create a dictionary whose keys are metamessages files and whose values are labels
    """
    labels_dict = {}
    for file in files:
        data_df = get_df_file(file)
        X, labels = to_input_output(data_df,
                                    scaler=scaler,
                                    columns_to_scale=columns_to_scale)
        labels_dict[file] = labels
    return labels_dict


def create_dataloaders_dict(datasets_dict, shuffle=False, batch_size=512):
    """
    Create a dictionary whose keys are metamessages files and whose values are dataloaders
    """
    dataloaders = {}
    for file in datasets_dict.keys():
        dataloaders[file] = DataLoader(datasets_dict[file],
                                       shuffle=shuffle,
                                       batch_size=batch_size)
    return dataloaders


def create_dataloaders_dict_from_directory(dataset_directory,
                                           scaler=0,
                                           device='cpu',
                                           force_batch_equal_1=0,
                                           shuffle=False):
    """
    Create a dictionary whose keys are metamessages files from a dataset directory and whose values are dataloaders
    """
    files = glob.glob(
        f"../datasets/{dataset_directory}/*.csv"
    ) 
    if scaler == 0:
        scaler = get_scaler_from_files(
            files, columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'])
    datasets_dict = create_datasets_dict(
        files,
        scaler=scaler,
        columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
        device=device)
    if force_batch_equal_1 == 1:
        total_loader = create_dataloaders_dict(datasets_dict,
                                               shuffle=shuffle,
                                               batch_size=1)
    else:
        total_loader = create_dataloaders_dict(datasets_dict,
                                               shuffle=shuffle,
                                               batch_size=config['batch_size'])
    return total_loader, scaler


def concat_dataloader_dict(dataloader_dict, shuffle=False):
    """
    Concatenate dataloaders from a dataloader dictionary into a one big dataloader
    """
    dataset_list = [loader.dataset for loader in dataloader_dict.values()]
    total_dataset = ConcatDataset(dataset_list)
    total_loader = DataLoader(total_dataset,
                              shuffle=shuffle,
                              batch_size=config['batch_size'])
    return total_loader


def get_df_real_df_pred(file, scaler, test_loader_dict, test_labels_dict,
                        model):
    """
    Get real and predicted dataframe used for evaluation
    """
    labels, labels_pred = get_output_used_for_evaluation(
        test_loader_dict[file], model)
    size = labels.size(0)
    df = get_df_file(
        file,
        scaler=scaler,
        columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
        enable_decorator=False)
    df_real = df.iloc[-size:]
    df_pred = df.iloc[
        -size:] 
    df_real.loc[:,'24'] = labels
    df_pred.loc[:,'24'] = labels_pred
    df_real = df_real.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    return df_real, df_pred


def evaluate(meta_message_label, meta_message_label_pred):
    """
    Returns "TN", "FP", "FN" or "TP"
    """
    if meta_message_label == 0 and meta_message_label_pred == 0:
        return "TN"
    elif meta_message_label == 0 and meta_message_label_pred == 1:
        return "FP"
    elif meta_message_label == 1 and meta_message_label_pred == 0:
        return "FN"
    elif meta_message_label == 1 and meta_message_label_pred == 1:
        return "TP"


class ANALYZE_FLIGHT:
    """
    Analyze flight: evaluation as well as alarm evaluation
    """
    def __init__(self, file, scaler, test_loader_dict, test_labels_dict,
                 model):
        self.file = file
        self.anomaly_score = None
        self.df, self.df_pred = get_df_real_df_pred(file, scaler,
                                                    test_loader_dict,
                                                    test_labels_dict, model)
        self.labels = self.df['24']  #.apply(int)
        self.t = self.df['7_8']
        self.t_pred = self.df_pred['7_8']
        self.labels_detected = self.df_pred['24']
        if 1 in self.df['24'].values:
            self.flight_attacked = 1
        else:
            self.flight_attacked = 0

        self.alarm_evaluated = self.evaluation_alarm()
        self.file_basename = os.path.basename(self.file)
    def launch_alarm_condition(self,
                               type="detector",
                               threshold=config['threshold_alarm']):
        """
        Returns 1 if alarm condition met
        """
        alarm_cumul_local = 0
        nb_messages_till_alarm_local = 0
        alarm = 0
        if type == "detector":
            labels = self.labels_detected
        elif type == "theoretical":
            labels = self.labels
        else:
            raise Exception(
                "Choose the alarm condition type (detector or theoretical) ")
        for out in labels:
            if out == 1:
                alarm_cumul_local = alarm_cumul_local + 1
            if alarm_cumul_local >= threshold:
                self.nb_messages_till_alarm = nb_messages_till_alarm_local + 1
                self.alarm_cumul = alarm_cumul_local
                alarm = 1
                break
            nb_messages_till_alarm_local = nb_messages_till_alarm_local + 1
        return alarm

    def evaluation_alarm(self, threshold=config['threshold_alarm']):
        self.alarm_pred = self.launch_alarm_condition(type="detector",
                                                      threshold=threshold)
        self.alarm_real = self.launch_alarm_condition(type="theoretical",
                                                      threshold=threshold)
        return {
            "detector": evaluate(self.flight_attacked, self.alarm_pred),
            "theoretical": evaluate(self.flight_attacked, self.alarm_real)
        }  #TN or ...

    def print_evaluated_alarm(self):
        print(
            f"alarm launching: ({self.file_basename})  detector:{self.alarm_evaluated['detector']} theoretical_detector:{self.alarm_evaluated['theoretical']} isattacked:{self.flight_attacked}"
        )


class ANALYZE_FLIGHTS:
    """
    Analyze multiple flights
    """
    def __init__(self, *args):
        #if args > 1 (list of Analyze_Flight objects)
        self.anomaly_score = []
        if isinstance(args[0], ANALYZE_FLIGHT):
            self.analyze_flight = {}
            for analyze_flight in args:
                self.analyze_flight[analyze_flight.file] = analyze_flight
            self.analyze_flight_list = self.analyze_flight.values
            self.add_members_after_args_verification()
        else:
            self.analyze_flight = {}
            for file in args[1].keys(
            ):  #args[2] is labels dict which does not contain small flights
                self.analyze_flight[file] = ANALYZE_FLIGHT(
                    file, args[0], args[1], args[2], args[3])
            self.analyze_flight_list = self.analyze_flight.values()
            self.add_members_after_args_verification()

    def add_members_after_args_verification(self):
        """
        Add members to the ANALYZE_FLIGHTS object after verification of the validity of arguments
        """
        self.labels = []  #total labels list
        self.labels_detected = []  #total labels list
        total_alarm_pred = []
        total_alarm_real = []
        total_flight_attacked = []
        for f in tqdm(self.analyze_flight_list):
            self.labels.extend(f.labels.tolist())
            self.labels_detected.extend(f.labels_detected.tolist())
            total_alarm_pred.extend([f.alarm_pred])
            total_alarm_real.extend([f.alarm_real])
            total_flight_attacked.extend([f.flight_attacked])
            #Uncomment next line to print which flights have been properly classified
            #f.print_evaluated_alarm()
        self.evaluation = evaluation_detection_metrics(self.labels,
                                                       self.labels_detected)
        self.evaluation_alarm_pred = evaluation_detection_metrics(
            total_flight_attacked, total_alarm_pred)
        self.evaluation_alarm_real = evaluation_detection_metrics(
            total_flight_attacked, total_alarm_real)
        self.evaluation_alarm_pred_vs_alarm_real = evaluation_detection_metrics(
            total_alarm_real, total_alarm_pred)

    def print_evaluation(self):
        print("\nEvaluation of classification of meta messages:\n")
        print_evaluation_detection_metrics(self.evaluation)
        print("\nPerformance of alarm: the performance of the classification of individual flights as containing any attack or not.\n")
        print("\nPerformance of alarm from predictions:\n")
        print_evaluation_detection_metrics(self.evaluation_alarm_pred)
        print("\nTheoretical alarm:\nThis is an alarm that depends on a prefect theoretical detector that can detect all anomalous meta messages. In other words this alarm is launched if the number of anomalies in a given flight is above the specified alarm threshold.\n")
        print("\nPerformance of theoretical alarm:\n")
        print_evaluation_detection_metrics(self.evaluation_alarm_real)
        print("\nPerformance of alarm from predictions compared to the performance of theoretical alarm:\n")
        print_evaluation_detection_metrics(
            self.evaluation_alarm_pred_vs_alarm_real)


def get_inputs_to_analyze_test_flights(force_batch_equal_1=0):
    """
    Get inputs to analyze test flights
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #scaler=0 to scale, None to not scale (good for anomaly detection)
    if config['scaling']:
        scaler = 0
    else:
        scaler = None
    train_loader_dict, scaler = create_dataloaders_dict_from_directory(
        config['train_directory'],
        scaler=scaler,
        force_batch_equal_1=0,
        device=device,
        shuffle=False)
    test_loader_dict, _ = create_dataloaders_dict_from_directory(
        config['test_directory'],
        scaler=scaler,
        force_batch_equal_1=force_batch_equal_1,
        device=device,
        shuffle=False)
    total_train_loader = concat_dataloader_dict(train_loader_dict,
                                                shuffle=False)
    model = get_model(total_train_loader, device=device, pretrained=1)

    train_files = glob.glob(f"../datasets/{config['train_directory']}/*.csv")
    test_files = glob.glob(f"../datasets/{config['test_directory']}/*.csv")
    test_labels_dict = create_labels_dict(
        test_files,
        scaler=scaler,
        columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
        device=device,
    )
    return scaler, test_loader_dict, test_labels_dict, model


def main_analyze_test_flights():
    scaler, test_loader_dict, test_labels_dict, model = get_inputs_to_analyze_test_flights(
    )
    analyze_flights = ANALYZE_FLIGHTS(scaler, test_loader_dict,
                                      test_labels_dict, model)
    analyze_flights.print_evaluation()   
    print("finished")
