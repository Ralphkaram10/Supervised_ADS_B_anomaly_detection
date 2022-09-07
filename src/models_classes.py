import torch
from torch import nn
import yaml
import math

#load the chosen config file
with open(".config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
class MV_LSTM_SUPERVISED(torch.nn.Module):
    """
    LSTM to predict labels
    """
    def __init__(self, n_features, seq_length, device):
        super(MV_LSTM_SUPERVISED, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 64  # number of hidden states
        self.n_hidden2 = 32
        self.n_layers = 1  # number of LSTM layers (stacked)
        self.linear_combine = torch.nn.Linear(self.seq_len * n_features + 1,
                                              self.seq_len * n_features)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True,
                                    bias=False)
        self.l_lstm2 = torch.nn.LSTM(input_size=self.n_hidden,
                                     hidden_size=self.n_hidden2,
                                     num_layers=1,
                                     batch_first=True)
        self.l_linear_supervised = torch.nn.Linear(
            self.n_hidden2 * self.seq_len, 1)
        self.relu = nn.ReLU()
        self.device = device

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers,
                                   batch_size,
                                   self.n_hidden,
                                   device=self.device)
        cell_state = torch.zeros(self.n_layers,
                                 batch_size,
                                 self.n_hidden,
                                 device=self.device)

        hidden_state2 = torch.zeros(self.n_layers,
                                    batch_size,
                                    self.n_hidden2,
                                    device=self.device)
        cell_state2 = torch.zeros(self.n_layers,
                                  batch_size,
                                  self.n_hidden2,
                                  device=self.device)

        self.hidden = (hidden_state, cell_state)
        self.hidden2 = (hidden_state2, cell_state2)

    def forward(self, x, time):
        batch_size, seq_len, _ = x.size()

        if config['future_time_as_input'] == 1:
            x = x.view(batch_size, -1)
            time = time.view(batch_size, -1)
            x = torch.cat((x, time), dim=1) #concat future timestamp to input
            x = self.linear_combine(x)
            x = self.relu(x)

        x = x.view(batch_size, seq_len, -1)

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        lstm_out2, self.hidden2 = self.l_lstm2(lstm_out, self.hidden2)

        x = lstm_out2.contiguous().view(batch_size, -1)
        x = self.l_linear_supervised(x)
        return x 

    def predict(self, x, time):
        self.init_hidden(x.size(0))
        return self.forward(x, time)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





class TRANSFORMER_SUPERVISED(nn.Module):
    def __init__(self, n_features, seq_length, device):#, n_features, seq_length, device
        super(TRANSFORMER_SUPERVISED, self).__init__()
        self.seq_len = seq_length
        self.linear_combine = torch.nn.Linear(self.seq_len * n_features + 1,
                                              self.seq_len * n_features)
        d_model=8 #needs to be even
        self.linear_to_posencoding=torch.nn.Linear(n_features,d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # self.embedding = nn.Embedding(vocab_size, n_features)
        self.layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, device=device)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=1)
        self.decoder = nn.Linear(self.seq_len * d_model, 1)
        self.relu = nn.ReLU()
        self.device = device
        
    def init_hidden(self, batch_size):
        """
        Used only for consistency in code 
        """
        pass

    def forward(self, x, time):    
        # X shape: [seq_len, batch_size]
        batch_size, seq_len, _ = x.size()

        if config['future_time_as_input'] == 1:
            x = x.view(batch_size, -1)
            time = time.view(batch_size, -1)
            x = torch.cat((x, time), dim=1)  #concat future timestamp to input
            x = self.linear_combine(x)
            x = self.relu(x)

        x = x.view(batch_size, seq_len, -1)
        x =self.linear_to_posencoding(x)
        x = torch.transpose(
            x, 0,
            1)  #transpose(x, 0, 1) in order to have (seq, batch, feature)
        # x = self.embedding(x)
        # print("Embedding size [seq_len, batch_size, n_features]")
        # print(x.shape)
        x = self.pos_encoder(x)
        # X shape: [seq_len, batch_size, n_features]
        #print(x.shape)
        x = self.transformer(x)
        # X shape: [seq_len, batch_size, n_features]
        
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(batch_size, -1)
        #print(x.size())
        x = self.decoder(x)
        return x
