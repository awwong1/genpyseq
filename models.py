"""Recurrent neural networks for Sequence Generation
"""
import logging
import json
import torch
import torch.nn as nn

logger = logging.getLogger("genpyseq")


class CharRNN(nn.Module):
    """Character level recurrent neural network
    """

    def __init__(self, num_characters,
                 hidden_size=128, recurrent_type="LSTM", recurrent_layers=1, recurrent_dropout=0,
                 use_cuda=False):
        super(CharRNN, self).__init__()

        self.use_cuda = use_cuda

        self.recurrent_type = recurrent_type.upper()
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.recurrent_dropoout = recurrent_dropout

        rn_kwargs = {
            "input_size": num_characters,
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

        decoder_units = 2 * hidden_size
        self.decoder = nn.Linear(decoder_units, num_characters)
        self.softmax = nn.LogSoftmax(dim=2)

        logger.info("{}".format(self))
        if self.use_cuda:
            self.cuda()

    def forward(self, inp_val, hidden):
        for_decoder, hidden = self.rnn(inp_val, hidden)
        # for_decoder = for_decoder.expand(self.recurrent_layers, 1, -1)
        if self.recurrent_type == "LSTM":
            for_decoder = torch.cat((for_decoder, hidden[0]), dim=2)
        else:
            for_decoder = torch.cat((for_decoder, hidden), dim=2)
        for_softmax = self.decoder(for_decoder)
        output = self.softmax(for_softmax)
        return output, hidden

    def init_hidden(self, batch_size):
        if self.recurrent_type == "LSTM":
            hidden = (torch.zeros(self.recurrent_layers,
                                  batch_size, self.hidden_size),
                      torch.zeros(self.recurrent_layers,
                                  batch_size, self.hidden_size))
            if self.use_cuda:
                return (hidden[0].cuda(), hidden[1].cuda())
            return hidden

        hidden = torch.zeros(self.recurrent_layers,
                             batch_size, self.hidden_size)
        if self.use_cuda:
            return hidden.cuda()
        return hidden

    def save(self, epoch=0, loss=0, interrupted=False):
        """Save the model state dictionary"""
        interrupt = ""
        if interrupted:
            interrupt = "-INTERRUPTED"
        model_path = ("./models/char{type}{hidden_size}-" +
                      "layer{layers}-drop{dropout}-loss{loss:.5f}-epoch{epoch:03d}{interrupt}.pt").format(
            type=self.recurrent_type,
            hidden_size=self.hidden_size,
            layers=self.recurrent_layers,
            dropout=self.recurrent_dropoout,
            epoch=epoch,
            loss=loss,
            interrupt=interrupt
        )
        torch.save(self.state_dict(), model_path)
        return model_path

    def save_progress(self, progress_dict):
        """Save the epoch progress dictionary"""
        progress_path = self.get_progress_path()
        with open(progress_path, "w") as f:
            json.dump(progress_dict, f)
        return progress_path

    def get_progress_path(self):
        return ("./models/char{type}{hidden_size}-" +
                "layer{layers}-drop{dropout}.json").format(
            type=self.recurrent_type,
            hidden_size=self.hidden_size,
            layers=self.recurrent_layers,
            dropout=self.recurrent_dropoout)

    def get_state_dict_path(self):
        model_path = ("./models/char{type}{hidden_size}-" +
                      "layer{layers}-drop{dropout}.pt").format(
            type=self.recurrent_type,
            hidden_size=self.hidden_size,
            layers=self.recurrent_layers,
            dropout=self.recurrent_dropoout,
        )
        return model_path


class TokenRNN(nn.Module):
    """Token level recurrent neural network
    """

    def __init__(self, num_token_types, num_token_literals,
                 hidden_size=128, recurrent_type="LSTM", recurrent_layers=1, recurrent_dropout=0,
                 use_cuda=False):
        super(TokenRNN, self).__init__()
        self.use_cuda = use_cuda

        self.recurrent_type = recurrent_type.upper()
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.recurrent_dropoout = recurrent_dropout

        type_rn_kwargs = {
            "input_size": num_token_types,
            "hidden_size": hidden_size,
            "num_layers": recurrent_layers,
            "dropout": recurrent_dropout,
        }
        literal_rn_kwargs = {
            "input_size": num_token_literals,
            "hidden_size": hidden_size,
            "num_layers": recurrent_layers,
            "dropout": recurrent_dropout,
        }

        if self.recurrent_type == "RNN":
            self.type_rnn = nn.RNN(**type_rn_kwargs)
            self.literal_rnn = nn.RNN(**literal_rn_kwargs)
        elif self.recurrent_type == "LSTM":
            self.type_rnn = nn.LSTM(**type_rn_kwargs)
            self.literal_rnn = nn.LSTM(**literal_rn_kwargs)
        elif self.recurrent_type == "GRU":
            self.type_rnn = nn.GRU(**type_rn_kwargs)
            self.literal_rnn = nn.GRU(**literal_rn_kwargs)
        else:
            raise "Invalid recurrent layer type: {}".format(recurrent_type)

        decoder_units = 2 * hidden_size
        self.type_decoder = nn.Linear(decoder_units, num_token_types)
        literal_decoder_units = decoder_units + num_token_types
        self.literal_decoder = nn.Linear(
            literal_decoder_units, num_token_literals)
        self.softmax = nn.LogSoftmax(dim=2)

        logger.info("{}".format(self))
        if self.use_cuda:
            self.cuda()

    def forward(self, type_inp_val, literal_inp_val, type_hidden, literal_hidden):
        for_type_decoder, type_hidden = self.type_rnn(
            type_inp_val, type_hidden)
        for_literal_decoder, literal_hidden = self.literal_rnn(
            literal_inp_val, literal_hidden)

        if self.recurrent_type == "LSTM":
            for_type_decoder = torch.cat(
                (for_type_decoder, type_hidden[0]), dim=2)
            for_literal_decoder = torch.cat(
                (for_literal_decoder, literal_hidden[0]), dim=2)
        else:
            for_type_decoder = torch.cat(
                (for_type_decoder, type_hidden), dim=2)
            for_literal_decoder = torch.cat(
                (for_literal_decoder, literal_hidden), dim=2)

        for_type_softmax = self.type_decoder(for_type_decoder)
        for_literal_decoder = torch.cat(
            (for_literal_decoder, for_type_softmax), dim=2)
        for_literal_softmax = self.literal_decoder(for_literal_decoder)

        type_output = self.softmax(for_type_softmax)
        literal_output = self.softmax(for_literal_softmax)
        return type_output, literal_output, type_hidden, literal_hidden

    def init_hidden(self, batch_size):
        if self.recurrent_type == "LSTM":
            type_hidden = (torch.zeros(self.recurrent_layers,
                                       batch_size, self.hidden_size),
                           torch.zeros(self.recurrent_layers,
                                       batch_size, self.hidden_size))
            literal_hidden = (torch.zeros(self.recurrent_layers,
                                          batch_size, self.hidden_size),
                              torch.zeros(self.recurrent_layers,
                                          batch_size, self.hidden_size))
            if self.use_cuda:
                return ((type_hidden[0].cuda(), type_hidden[1].cuda()),
                        (literal_hidden[0].cuda(), literal_hidden[0].cuda()))
            return type_hidden, literal_hidden

        type_hidden = torch.zeros(self.recurrent_layers,
                                  batch_size, self.hidden_size)
        literal_hidden = torch.zeros(
            self.recurrent_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            return type_hidden.cuda(), literal_hidden.cuda()
        return type_hidden, literal_hidden

    def save(self, epoch=0, loss=0, interrupted=False):
        """Save the model state dictionary"""
        interrupt = ""
        if interrupted:
            interrupt = "-INTERRUPTED"
        model_path = ("./models/token{type}{hidden_size}-" +
                      "layer{layers}-drop{dropout}-loss{loss:.5f}-epoch{epoch:03d}{interrupt}.pt").format(
            type=self.recurrent_type,
            hidden_size=self.hidden_size,
            layers=self.recurrent_layers,
            dropout=self.recurrent_dropoout,
            epoch=epoch,
            loss=loss,
            interrupt=interrupt
        )
        torch.save(self.state_dict(), model_path)
        return model_path

    def save_progress(self, progress_dict):
        """Save the epoch progress dictionary"""
        progress_path = self.get_progress_path()
        with open(progress_path, "w") as f:
            json.dump(progress_dict, f)
        return progress_path

    def get_progress_path(self):
        return ("./models/token{type}{hidden_size}-" +
                "layer{layers}-drop{dropout}.json").format(
            type=self.recurrent_type,
            hidden_size=self.hidden_size,
            layers=self.recurrent_layers,
            dropout=self.recurrent_dropoout)

    def get_state_dict_path(self):
        model_path = ("./models/token{type}{hidden_size}-" +
                      "layer{layers}-drop{dropout}.pt").format(
            type=self.recurrent_type,
            hidden_size=self.hidden_size,
            layers=self.recurrent_layers,
            dropout=self.recurrent_dropoout,
        )
        return model_path
