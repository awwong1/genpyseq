#!/usr/bin/env python3
import argparse
import torch
from torchvision import transforms

from datasets import CHARACTERS, CharDataset, CharSequenceToTensor
from models import CharRNN
from train import train_full

DEFAULT_RECURRENT_TYPE = "LSTM"
DEFAULT_RECURRENT_HIDDEN_SIZE = 128
DEFAULT_RECURRENT_LAYERS = 1
DEFAULT_RECURRENT_DROPOUT = 0.0
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 200
DEFAULT_BATCH_SIZE = 256
DEFAULT_WINDOW_SIZE = 20
DEFAULT_USE_CUDA = torch.cuda.is_available()


def main(
    train=None,
    window_size=DEFAULT_WINDOW_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    use_cuda=DEFAULT_USE_CUDA,
    learning_rate=DEFAULT_LEARNING_RATE,
    num_epochs=DEFAULT_NUM_EPOCHS,
    recurrent_type=DEFAULT_RECURRENT_TYPE,
    hidden_size=DEFAULT_RECURRENT_HIDDEN_SIZE,
    recurrent_layers=DEFAULT_RECURRENT_LAYERS,
    recurrent_dropout=DEFAULT_RECURRENT_DROPOUT
):
    if train == "char":
        # Create the neural network structure
        n_chars = len(CHARACTERS)
        nn = CharRNN(
            n_chars, n_chars, hidden_size=hidden_size,
            recurrent_type=recurrent_type, recurrent_layers=recurrent_layers,
            recurrent_dropout=recurrent_dropout)

        device = torch.device("cpu")
        if use_cuda:
            nn.cuda()
            device = torch.device("cuda")

        # Instantiate the dataset
        ds = CharDataset(
            max_window_size=window_size,
            transform=transforms.Compose(
                [CharSequenceToTensor(device=device), ]))

        # Train our model
        train_full(nn, ds, learning_rate=learning_rate,
                   n_epochs=num_epochs, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train", help="train model",
        type=str,
        choices=["char"])
    parser.add_argument(
        "--window-size",
        help="sequence window upper bound size for data (default: {})".format(
            DEFAULT_WINDOW_SIZE),
        type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--batch-size",
        help="batch size of data (default: {})".format(DEFAULT_BATCH_SIZE),
        type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--use-cuda", help="should model use cuda (default: {})".format(DEFAULT_USE_CUDA),
        type=bool, default=DEFAULT_USE_CUDA)
    parser.add_argument(
        "--learning-rate",
        help="train optimizer learning rate (default: {})".format(
            DEFAULT_LEARNING_RATE),
        type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--num-epochs",
        help="train number of epochs (default: {})".format(DEFAULT_NUM_EPOCHS),
        type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument(
        "--recurrent-type",
        help="type of recurrent network (default: {})".format(
            DEFAULT_RECURRENT_TYPE),
        choices=["LSTM", "RNN", "GRU"],
        type=str, default=DEFAULT_RECURRENT_TYPE)
    parser.add_argument(
        "--hidden-size",
        help="size of recurrent hidden vector (default: {})".format(
            DEFAULT_RECURRENT_HIDDEN_SIZE),
        type=int, default=DEFAULT_RECURRENT_HIDDEN_SIZE)
    parser.add_argument(
        "--recurrent-layers",
        help="number of recurrent layers (default: {})".format(
            DEFAULT_RECURRENT_LAYERS),
        type=int, default=DEFAULT_RECURRENT_LAYERS)
    parser.add_argument(
        "--recurrent-dropout",
        help="ratio of recurrent units to drop (default: {})".format(
            DEFAULT_RECURRENT_DROPOUT),
        type=float, default=DEFAULT_RECURRENT_DROPOUT)

    args = parser.parse_args()
    main(
        train=args.train,
        window_size=args.window_size,
        batch_size=args.batch_size,
        use_cuda=args.use_cuda,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        recurrent_type=args.recurrent_type,
        hidden_size=args.hidden_size,
        recurrent_layers=args.recurrent_layers,
        recurrent_dropout=args.recurrent_dropout
    )
