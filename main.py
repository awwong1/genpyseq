#!/usr/bin/env python3
import sys
import logging
import argparse
import torch
from torchvision import transforms

from datasets import CharDataset, CharSequenceToTensor, TokenDataset, TokenSequenceToTensor
from models import CharRNN, TokenRNN
from train import CharTrainer, TokenTrainer
from generate import generate_charseq, generate_tokenseq

logger = logging.getLogger("genpyseq")

DEFAULT_REPRESENTATION = "char"
DEFAULT_RECURRENT_TYPE = "LSTM"
DEFAULT_RECURRENT_HIDDEN_SIZE = 300
DEFAULT_RECURRENT_LAYERS = 1
DEFAULT_RECURRENT_DROPOUT = 0.0
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 2000
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_WINDOW_SIZE = None
DEFAULT_PRINT_EVERY_ITER = 179
DEFAULT_DISABLE_CUDA = False
DEFAULT_TEMPERATURE = None
DEFAULT_MAX_GEN_LEN = 1000
DEFAULT_LOG_LEVEL = "INFO"


def main(
    representation,
    train=None,
    generate=None,
    data_path="./data/charseqs.json",
    temperature=DEFAULT_TEMPERATURE,
    max_generate_len=DEFAULT_MAX_GEN_LEN,
    generator_prime_str=CharDataset.FILE_START,
    window_size=DEFAULT_WINDOW_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    disable_cuda=DEFAULT_DISABLE_CUDA,
    learning_rate=DEFAULT_LEARNING_RATE,
    num_epochs=DEFAULT_NUM_EPOCHS,
    patience=DEFAULT_PATIENCE,
    recurrent_type=DEFAULT_RECURRENT_TYPE,
    hidden_size=DEFAULT_RECURRENT_HIDDEN_SIZE,
    recurrent_layers=DEFAULT_RECURRENT_LAYERS,
    recurrent_dropout=DEFAULT_RECURRENT_DROPOUT,
    print_every_iter=DEFAULT_PRINT_EVERY_ITER,
    log_level=DEFAULT_LOG_LEVEL,
):
    # https://github.com/pytorch/pytorch/issues/13775
    torch.multiprocessing.set_start_method("spawn")

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(log_level)

    use_cuda = torch.cuda.is_available()
    if disable_cuda:
        use_cuda = False

    logger.info("Code Representation: {}".format(representation))

    if representation == "char":
        dataset = CharDataset(max_window_size=window_size,
                              data_path=data_path,
                              transform=transforms.Compose(
                                  [CharSequenceToTensor(cuda=use_cuda), ]))

        logger.info("Constructing the neural network architecture...")
        nn = CharRNN(
            len(dataset.CHARACTERS), hidden_size=hidden_size,
            recurrent_type=recurrent_type, recurrent_layers=recurrent_layers,
            recurrent_dropout=recurrent_dropout, use_cuda=use_cuda)
        if train:
            CharTrainer.train_full(nn, dataset, max_window_size=window_size,
                                   learning_rate=learning_rate,
                                   n_epochs=num_epochs,
                                   patience_threshold=patience,
                                   batch_size=batch_size,
                                   print_every=print_every_iter,
                                   use_cuda=use_cuda)
        elif generate:
            generate_charseq(nn, prime_str=generator_prime_str,
                             max_window_size=window_size,
                             max_generate_len=max_generate_len,
                             temperature=temperature)
        else:
            logger.info("train or generate not specified? (See --help)")
    elif representation == "token":
        dataset = TokenDataset(max_window_size=window_size,
                               data_path=data_path,
                               transform=transforms.Compose(
                                   [TokenSequenceToTensor(cuda=use_cuda), ]))

        logger.info("Constructing the neural network architecture...")
        nn = TokenRNN(
            len(dataset.INT2TOKENTYPE), len(dataset.INT2LITERAL),
            hidden_size=hidden_size, recurrent_type=recurrent_type,
            recurrent_layers=recurrent_layers,
            recurrent_dropout=recurrent_dropout, use_cuda=use_cuda)
        if train:
            TokenTrainer.train_full(nn, dataset, max_window_size=window_size,
                                    learning_rate=learning_rate,
                                    n_epochs=num_epochs,
                                    patience_threshold=patience,
                                    batch_size=batch_size,
                                    print_every=print_every_iter,
                                    use_cuda=use_cuda)
        elif generate:
            generate_tokenseq(nn, prime_str=generator_prime_str,
                              max_window_size=window_size,
                              max_generate_len=max_generate_len,
                              temperature=temperature)
        else:
            logger.info("train or generate not specified? (See --help)")


class ArgparseRange(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __str__(self):
        return "[{}, {}]".format(self.start, self.end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generative models for Python Source Code")

    parser.add_argument(
        "representation",
        help="Python source code data representation",
        type=str, choices=["char", "token"])
    parser.add_argument(
        "--train", help="train model",
        action="store_true", default=False)
    parser.add_argument(
        "--generate", help="generate source code",
        action="store_true", default=False)
    parser.add_argument(
        "--data-path", help="path to json data file of sequences",
        action="store", default="./data/charseqs.json"
    )

    parser.add_argument(
        "--temperature",
        help="value to divide over log probabilities before code generation (default: {})".format(
            DEFAULT_TEMPERATURE),
        type=float, choices=[ArgparseRange(0.0, 1.0)])
    parser.add_argument(
        "--max-generate-len",
        help="maximum number of elements to generate (default: {})".format(
            DEFAULT_MAX_GEN_LEN),
        type=int, default=DEFAULT_MAX_GEN_LEN)
    parser.add_argument(
        "--generator-prime-str",
        help="string to prime generator hidden state with (default: {})".format(
            repr("")),
        type=str, default="")

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
        "--patience",
        help="Early stop if no loss improvement this many epochs (default: {})".format(
            DEFAULT_PATIENCE),
        type=int, default=DEFAULT_PATIENCE)
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
    # Only one recurrent layer currently supported
    # parser.add_argument(
    #     "--recurrent-layers",
    #     help="number of recurrent layers (default: {})".format(
    #         DEFAULT_RECURRENT_LAYERS),
    #     type=int, default=DEFAULT_RECURRENT_LAYERS)
    # parser.add_argument(
    #     "--recurrent-dropout",
    #     help="ratio of recurrent units to drop (default: {})".format(
    #         DEFAULT_RECURRENT_DROPOUT),
    #     type=float, default=DEFAULT_RECURRENT_DROPOUT)
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
        data_path=args.data_path,
        temperature=args.temperature,
        max_generate_len=args.max_generate_len,
        generator_prime_str=args.generator_prime_str,
        window_size=args.window_size,
        batch_size=args.batch_size,
        disable_cuda=args.disable_cuda,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        recurrent_type=args.recurrent_type,
        hidden_size=args.hidden_size,
        # recurrent_layers=args.recurrent_layers,
        # recurrent_dropout=args.recurrent_dropout,
        print_every_iter=args.print_every_iter,
        log_level=args.log_level
    )
