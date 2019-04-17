#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import torch
from torchvision import transforms

from datasets import CHARACTERS, CharDataset, CharSequenceToTensor
from models import CharRNN
from train import train_full

logger = logging.getLogger("genpyseq")

DEFAULT_REPRESENTATION = "code"
DEFAULT_RECURRENT_TYPE = "LSTM"
DEFAULT_RECURRENT_HIDDEN_SIZE = 300
DEFAULT_RECURRENT_LAYERS = 1
DEFAULT_RECURRENT_DROPOUT = 0.0
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 200
DEFAULT_BATCH_SIZE = 1
DEFAULT_WINDOW_SIZE = None
DEFAULT_PRINT_EVERY_ITER = 179
DEFAULT_DISABLE_CUDA = False

DEFAULT_LOG_LEVEL = "INFO"

def main(
    representation,
    train=None,
    generate=None,
    window_size=DEFAULT_WINDOW_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    disable_cuda=DEFAULT_DISABLE_CUDA,
    learning_rate=DEFAULT_LEARNING_RATE,
    num_epochs=DEFAULT_NUM_EPOCHS,
    recurrent_type=DEFAULT_RECURRENT_TYPE,
    hidden_size=DEFAULT_RECURRENT_HIDDEN_SIZE,
    recurrent_layers=DEFAULT_RECURRENT_LAYERS,
    recurrent_dropout=DEFAULT_RECURRENT_DROPOUT,
    print_every_iter=DEFAULT_PRINT_EVERY_ITER,
    log_level=DEFAULT_LOG_LEVEL
):
    # https://github.com/pytorch/pytorch/issues/13775
    torch.multiprocessing.set_start_method("spawn")

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(log_level)

    use_cuda = torch.cuda.is_available()
    if disable_cuda:
        use_cuda = False

    if representation == "char":
        # Create the neural network structure
        logger.info("Creating the neural network architecture...")
        n_chars = len(CHARACTERS)
        nn = CharRNN(
            n_chars, n_chars, hidden_size=hidden_size,
            recurrent_type=recurrent_type, recurrent_layers=recurrent_layers,
            recurrent_dropout=recurrent_dropout, use_cuda=use_cuda)

        if use_cuda:
            nn.cuda()

        if train:
            # Instantiate the dataset
            logger.info("Initializing the dataset...")
            ds = CharDataset(
                max_window_size=window_size,
                transform=transforms.Compose(
                    [CharSequenceToTensor(cuda=use_cuda), ]))

            # Train our model
            logger.info("Training the neural network...")
            train_full(nn, ds, learning_rate=learning_rate,
                    n_epochs=num_epochs, batch_size=batch_size,
                    print_every=print_every_iter)
        elif generate:
            # Load our model
            logger.info("Loading the model weights...")
            path = nn.get_state_dict_path()
            if not os.path.isfile(path):
                raise FileNotFoundError("Model does not exist at {}".format(path))
            nn.load_state_dict(torch.load(path))
            nn.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generative models for Python Source Code")

    parser.add_argument(
        "representation",
        help="Python source code data representation",
        type=str, choices=["char"])
    parser.add_argument(
        "--train", help="train model",
        action="store_true", default=False)
    parser.add_argument(
        "--generate", help="generate source code",
        action="store_true", default=False)

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
        "--disable-cuda", help="use cpu for model device",
        action="store_true",
        default=DEFAULT_DISABLE_CUDA)
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
    parser.add_argument(
        "--print-every-iter",
        help="print training progress every # batch iterations (default: {})".format(
            DEFAULT_PRINT_EVERY_ITER),
        type=int, default=DEFAULT_PRINT_EVERY_ITER)
    parser.add_argument(
        "-l", "--log", dest="log_level",
        help="set the logging level (default: {})".format(DEFAULT_LOG_LEVEL),
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL)

    args = parser.parse_args()
    main(
        args.representation,
        train=args.train,
        generate=args.generate,
        window_size=args.window_size,
        batch_size=args.batch_size,
        disable_cuda=args.disable_cuda,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        recurrent_type=args.recurrent_type,
        hidden_size=args.hidden_size,
        recurrent_layers=args.recurrent_layers,
        recurrent_dropout=args.recurrent_dropout,
        print_every_iter=args.print_every_iter,
        log_level=args.log_level
    )
