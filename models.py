"""Recurrent neural networks for Sequence Generation
"""
import json
import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """Character level recurrent neural network
    """

    def __init__(self, input_size, output_size,
                 hidden_size=128, recurrent_type="LSTM", recurrent_layers=1, recurrent_dropout=0):
        super(CharRNN, self).__init__()

        self.recurrent_type = recurrent_type.upper()
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.recurrent_dropoout = recurrent_dropout

        rn_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": recurrent_layers,
            "dropout": recurrent_dropout,
        }

        if self.recurrent_type == "RNN":
            self.rnn = nn.RNN(**rn_kwargs)
        elif self.recurrent_type == "LSTM":
            self.rnn = nn.LSTM(**rn_kwargs)
        elif self.recurrent_type == "GRU":
            self.rnn = nn.GRU(**rn_kwargs)
        else:
            raise "Invalid recurrent layer type: {}".format(recurrent_type)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, inp_val, hidden):
        for_decoder, hidden = self.rnn(inp_val, hidden)
        for_softmax = self.decoder(for_decoder)
        output = self.softmax(for_softmax)
        return output, hidden

    def init_hidden(self, batch_size, device=torch.device("cpu")):
        if self.recurrent_type == "LSTM":
            return (
                torch.zeros(self.recurrent_layers, batch_size,
                            self.hidden_size, device=device),
                torch.zeros(self.recurrent_layers, batch_size,
                            self.hidden_size, device=device)
            )
        return torch.zeros(self.recurrent_layers, batch_size, self.hidden_size, device=device)

    def save(self, epoch=0, loss=0, interrupted=False):
        """Save the model state dictionary"""
        interrupt = ""
        if interrupted:
            interrupt = "-INTERRUPTED"
        model_path = ("./models/char{type}{hidden_size}-" +
                      "layer{layers}-drop{dropout}-loss{loss}-epoch{epoch:03d}{interrupt}").format(
            type=self.recurrent_type,
            hidden_size=self.hidden_size,
            layers=self.recurrent_layers,
            dropout=self.recurrent_dropoout,
            epoch=epoch,
            loss=loss,
            interrupt=interrupt
        )
        torch.save(self.state_dict(), model_path)

    def save_progress(self, progress_dict):
        """Save the epoch progress dictionary"""
        progress_path = ("./models/char{type}{hidden_size}-" +
                         "layer{layers}-drop{dropout}.json").format(
            type=self.recurrent_type,
            hidden_size=self.hidden_size,
            layers=self.recurrent_layers,
            dropout=self.recurrent_dropoout)
        with open(progress_path, "w") as f:
            json.dump(progress_dict, f)