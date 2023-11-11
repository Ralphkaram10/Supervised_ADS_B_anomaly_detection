import glob
import copy
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn
import numpy as np
import math
from torch.utils.data import Dataset, ConcatDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
from torch.utils.data import DataLoader

import yaml
import models_classes

#load the chosen config file
with open(".config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


def to_diff(func):
    """
    Decorator to the following get_df_file function that applies difference on consecutive metamessages
    """
    def inner(file,
              scaler=None,
              columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
              enable_decorator=True):
        df = func(file, scaler=scaler, columns_to_scale=columns_to_scale)
        if enable_decorator:
            not_difference_columns = ['24']
            df.loc[:, ~df.columns.
                   isin(not_difference_columns)] = df.loc[:, ~df.columns.isin(
                       not_difference_columns)].diff(periods=1)
            df = df.dropna(how='any')  #because of difference
        return df

    return inner


@conditional_decorator(to_diff, config['to_diff_cond'])
def get_df_file(file,
                scaler=None,
                columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
                enable_decorator=True):
    """
    Get csv meta message file as dataframe
    """
    df = pd.read_csv(file)
    df = df.drop(['17'], axis=1)
    #suppress counter (column 0)
    df = df.loc[:, df.columns != '0']
    #remove bad characters
    df = df[df != '�']
    df = df.astype(float)
    return df


def get_df_files(files):
    """
    Get csv meta message files as one concatenated dataframe
    """
    first = True
    for file in tqdm(files):
        if first:
            df = get_df_file(file)
            first = False
        else:
            df2 = get_df_file(file)
            #concat from different files
            df = pd.concat([df, df2])
    return df


def get_scaler_from_files(files,
                          columns_to_scale=[
                              '7_8', '4', '12', '13', '14', '15', '16'
                          ]):
    """
    Fit scaler to data from files and get scaler
    """
    df = get_df_files(files)
    scaler = MinMaxScaler()
    scaler.fit(df[columns_to_scale])
    return scaler


def split_sequences(sequences, n_steps, threshold=0):
    """
    Split target sequence into samples (To have one label per window)
    """
    label = []
    for i in tqdm(range(len(sequences))):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        last_col_index = sequences.shape[-1]
        _, seq_label = sequences[i:end_ix, 0:last_col_index -
                                 1].tolist(), sequences[i:end_ix,
                                                        last_col_index -
                                                        1].tolist()
        seq_label = (seq_label.count(1) / len(seq_label))
        if seq_label > threshold:
            seq_label = 1
        else:
            seq_label = 0
        label.append(seq_label)
    label = torch.as_tensor(label)
    return label.view(-1, 1)


def to_input_output(
        sequences,
        threshold=0,
        scaler=None,
        columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16']):
    """
    Get input and output tensors from dataframe
    """
    n_steps = config['lookback']
    if scaler is not None:
        sequences[columns_to_scale] = scaler.transform(
            sequences[columns_to_scale])
    sequences = torch.tensor(sequences.values)
    last_col_index = sequences.shape[-1]
    X = sequences[:, 0:last_col_index - 1].tolist()
    if config['label_for_metamessage_or_window'] == 0:
        label = sequences[n_steps - 1:, last_col_index - 1]
    elif config['label_for_metamessage_or_window'] == 1:
        label = split_sequences(sequences, n_steps, threshold=threshold)
    return torch.as_tensor(X), torch.as_tensor(label)


class TimeseriesSupervised(Dataset):
    """
    Torch Dataset class used for supervised anomaly detection
    """
    def __init__(self, X, y, device):
        self._X = X.float()
        self._y = y.float()
        if torch.cuda.is_available():
            self._X = self._X.to(device)
            self._y = self._y.to(device)
        self.n_steps = config['lookback']

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X.float()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y.float()

    def __getitem__(self, idx):
        return self._X[idx:idx + self.n_steps], self._X[
            idx + self.n_steps,
            0], self._y[idx]  #input, future time input, label

    def __len__(self):
        return self.X.__len__() - self.n_steps


def get_dataset_from_dataframe(
        data_df,
        scaler=None,
        columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
        datasets={},
        device=0):
    """
    Get dataset from dataframe
    """
    X, label = to_input_output(data_df,
                               scaler=scaler,
                               columns_to_scale=columns_to_scale)
    if len(X) > config['lookback']:
        dataset = TimeseriesSupervised(X, label, device)
        return dataset
    else:
        return None


def flight_is_too_small(length):
    """
    Returns 1 if flight is too small and 0 otherwise
    Note: flights too small for alarm can still be used for training and are not considered too small in the case of training a new model
    """
    if length < config['lookback']:
        print(f"flight too small for lookback, len={length}")
        return 1
    elif length < config['threshold_alarm'] and config['pretrained'] == 1:
        print(f"flight too small for Threshold of alarm, len={length}")
        return 1
    else:
        return 0


def remove_small_flights_from_dict(create_dict):
    """
    Decorator that removes small flights from dictionary whose keys are files and whose values are datasets or labels
    """
    def inner(files,
              scaler=None,
              columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
              device=0):
        dict = create_dict(files,
                           scaler=scaler,
                           columns_to_scale=columns_to_scale,
                           device=device)
        for file in files:
            n = dict.get(file, None)
            if n is not None:
                l = len(n)
                flight_size = l + config['lookback'] - 1
                if flight_is_too_small(flight_size):
                    del dict[file]
        return dict

    return inner


@remove_small_flights_from_dict
def create_datasets_dict(
        files,
        scaler=None,
        columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
        device=0):
    """
    Create a dictionary whose keys are metamessages files and whose values are torch datasets
    """
    datasets = {}
    if config['concatenated_df'] == 0:
        for file in files:
            data_df = get_df_file(file)
            dataset = get_dataset_from_dataframe(
                data_df,
                scaler=scaler,
                columns_to_scale=columns_to_scale,
                datasets=datasets,
                device=device)
            if dataset is not None:
                datasets[file] = dataset
    elif config['concatenated_df'] == 1:
        data_df = get_df_files(files)
        dataset = get_dataset_from_dataframe(data_df,
                                             scaler=scaler,
                                             columns_to_scale=columns_to_scale,
                                             datasets=datasets,
                                             device=device)
        if dataset is not None:
            datasets["combined_files"] = dataset
    return datasets


def predict_dataloader(data_loader, model):
    """
    Predict outputs of dataloader
    """
    first = 1
    for data, time, target in data_loader:
        model.init_hidden(data.size(0))
        if first == 1:
            outputs = model.forward(data, time)
            first = 0
        else:
            to_output = model.forward(data, time)
            outputs = torch.cat((outputs, to_output), 0)
    outputs = torch.where(outputs > 0.5, 1, 0)
    return outputs


def get_real_targets(data_loader):
    """
    Get real outputs of dataloader
    """
    first = 1
    for data, time, target in data_loader:
        if first == 1:
            outputs = target
            first = 0
        else:
            outputs = torch.cat((outputs, target), 0)
    outputs = torch.where(outputs > 0.5, 1, 0)
    return outputs


def get_data_loader(dataset_directory,
                    scaler=0,
                    device='cpu',
                    shuffle=False,
                    batch_size=config['batch_size']):
    """
    Get dataloader using dataset directory as input
    """
    files = glob.glob(
        f"../datasets/{dataset_directory}/*.csv"
    ) 
    if scaler == 0:
        scaler = get_scaler_from_files(
            files, columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'])
    dataset_dict = create_datasets_dict(
        files,
        scaler=scaler,
        columns_to_scale=['7_8', '4', '12', '13', '14', '15', '16'],
        device=device)
    total_dataset = ConcatDataset(dataset_dict.values())
    total_loader = DataLoader(total_dataset,
                              shuffle=shuffle,
                              batch_size=batch_size)
    # #to print shapes
    # dataiter = iter(total_loader)
    # input, time, output = dataiter.next()
    # print(f"nb_batches={len(total_dataset)}")
    # print(f"input={input.shape}\n")
    # print(f"time={time.shape}\n")
    # print(f"output={output.shape}\n")

    return total_loader, scaler


def train_model(model, total_train_loader, device='cpu', PATH=None):
    """
    Train model using total train loader as input
    """
    loss_dict = {"mse": torch.nn.MSELoss(), "mae": torch.nn.L1Loss()}
    optimizer_dict = {
        "adam": torch.optim.Adam(model.parameters(), lr=1e-3),
        "sgd": torch.optim.SGD(model.parameters(), lr=1e-3)
    }
    loss_function = loss_dict[config['loss_function']]
    optimizer = optimizer_dict[config['optimizer']]
    if torch.cuda.is_available():
        model.to(device)
    model.train()

    #if the number of dimensions is equal to 1 then it is the supervised case
    dataiter = iter(total_train_loader)
    input, time, output = dataiter.next()
    dim_number = len(list(output.size()))
    for t in tqdm(range(config['epochs'])):
        for data, time, target in total_train_loader:
            optimizer.zero_grad()  #necessary
            model.init_hidden(data.size(0))
            y_pred = model.forward(data, time)
            loss = loss_function(y_pred.view((-1, 1)), target.view((-1, 1)))
            loss.backward()
            optimizer.step()
        print("epoch : ", t, " loss=", loss)
    torch.save(model.state_dict(), PATH)
    return model


def get_model(total_train_loader,
              device='cpu',
              pretrained=config['pretrained']):
    """
    Train or load a pretrained model using total train loader as input
    """
    nb_features = total_train_loader.dataset.datasets[0].X.shape[1]
    if config['model_class'] == "MV_LSTM_SUPERVISED":
        model = models_classes.MV_LSTM_SUPERVISED(nb_features,
                                                  config['lookback'], device)
    elif config['model_class'] == "TRANSFORMER_SUPERVISED":
        model = models_classes.TRANSFORMER_SUPERVISED(nb_features,
                                                      config['lookback'],
                                                      device)
    if config['to_diff_cond'] == 1:
        diff = "_DIFF"

    else:
        diff = ""
    PATH = f"../models/{config['model_class']}{diff}_epochs={config['epochs']}_batchsize={config['batch_size']}_lookback={config['lookback']}_optim={config['optimizer']}_traindir={config['train_directory']}.pt"
    model_untrained = copy.deepcopy(model)
    print(f"\nModel used:\n\n{model}")
    if pretrained == 0:
        train_model(model, total_train_loader, device=device, PATH=PATH)
    model_untrained.load_state_dict(torch.load(PATH,map_location=device))
    if torch.cuda.is_available():
        model_untrained.to(device)
    return model_untrained


def get_output_used_for_evaluation(total_loader, model):
    """
    Get output used for evaluation
    """
    model.eval()
    with torch.no_grad():
        labels_pred = predict_dataloader(total_loader,
                                         model)  #model.predict(X)
        labels_pred = labels_pred.cpu()
        labels = get_real_targets(
            total_loader)  #necessary because there is batches
        labels = labels.cpu()
    return labels, labels_pred


def evaluation_detection_metrics(labels, labels_detected):
    """
    Get evaluation detection metrics
    """
    mse = mean_squared_error(labels, labels_detected)
    mae = mean_absolute_error(labels, labels_detected)
    rmse = sqrt(mse)
    conf_matrix = confusion_matrix(labels, labels_detected)
    if len(conf_matrix) > 1:
        precision = (conf_matrix[1, 1] /
                     (conf_matrix[1, 1] + conf_matrix[0, 1])) * 100
        recall = (conf_matrix[1, 1] /
                  (conf_matrix[1, 1] + conf_matrix[1, 0])) * 100
        fscore = (2 * precision * recall) / (precision + recall)

        TPR = (conf_matrix[1, 1] /
               (conf_matrix[1, 1] + conf_matrix[1, 0])) * 100
        FPR = (conf_matrix[0, 1] /
               (conf_matrix[0, 1] + conf_matrix[0, 0])) * 100
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'conf_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'TPR': TPR,
            'FPR': FPR
        }
    else:
        return 0


def print_evaluation_detection_metrics(evaluation):
    """
    Print evaluation detection metrics
    """
    classifier_evaluation = pd.DataFrame(
        columns=['conf_matrix', 'precision', 'recall', 'fscore'])
    if evaluation == 0:
        print("there is no attacks to be detected thus cannot be evaluated")
    else:
        row_dict = {
            'conf_matrix': evaluation['conf_matrix'],
            'precision': evaluation['precision'],
            'recall': evaluation['recall'],
            'fscore': evaluation['fscore']
        }
        test_mse = evaluation['mse']
        row_df = pd.DataFrame([row_dict])
        classifier_evaluation = pd.concat([classifier_evaluation, row_df],ignore_index=True)
        print('test_mse=', test_mse)
        print('classifier evaluation')
        print(classifier_evaluation)
        print('TPR=', evaluation['TPR'])
        print('FPR=', evaluation['FPR'])
        return classifier_evaluation


def main_training_testing_models():
    """
    Train (or use pretrained model) to predict window label or future label  and evaluate these predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #scaler=0 to scale, None to not scale
    if config['scaling']:
        scaler = 0
    else:
        scaler = None
    total_train_loader, scaler = get_data_loader(config['train_directory'],
                                                 scaler=scaler,
                                                 device=device,
                                                 shuffle=True)
    total_test_loader, _ = get_data_loader(config['test_directory'],
                                           scaler=scaler,
                                           device=device,
                                           shuffle=False,
                                           batch_size=config['batch_size'])
    model = get_model(total_train_loader,
                      device=device,
                      pretrained=config['pretrained'])
    labels, labels_pred = get_output_used_for_evaluation(
        total_test_loader, model)
    print("\nEvaluation of classification of meta messages:\n")
    print_evaluation_detection_metrics(
        evaluation_detection_metrics(labels, labels_pred))

    print("finished")
