#!/usr/bin/env python3
import logging
import os
from math import floor
from datetime import datetime
from statistics import mean
import torch
from torch.optim import Adam
from torch.nn import NLLLoss
from torchvision import transforms

from datasets import batch_collate_pairs, CharDataset, CharSequenceToTensor

logger = logging.getLogger("genpyseq")


def train_step(
        nn, target_tensor, input_tensor, criterion,
        optimizer=None, eval_only=False):
    """Perform one step of training for an input/target tensor pair.
    """
    window_size, batch_size, _ = input_tensor.size()
    hidden = nn.init_hidden(batch_size)

    nn.zero_grad()
    loss = 0
    match = []
    for i in range(window_size):
        output, hidden = nn(input_tensor.narrow(0, i, 1), hidden)
        _, target_character = target_tensor[i].topk(1)
        _, predict_character = output.topk(1)

        batch_match = []
        for batch_item_match in target_character.view(-1) == predict_character.view(-1):
            match.append(batch_item_match.item())
            batch_match.append(batch_item_match.item())

        loss += criterion(output.view(batch_size, -1),
                          target_character.view(-1))

    accuracy = mean(match)

    if not eval_only:
        loss.backward()
        optimizer.step()

    return accuracy, loss.item() / window_size


def evaluate_step(nn, target_tensor, input_tensor, criterion):
    return train_step(nn, target_tensor, input_tensor, criterion, eval_only=True)


def train_epoch(
        nn, train_dl, criterion, lr=0.001,
        start_time=datetime.now(), print_every=None):
    nn = nn.train()
    optimizer = Adam(nn.parameters(), lr=lr)

    train_losses = []
    for batch in train_dl:
        batch_input_tensor, batch_target_tensor = batch
        accuracy, train_loss = train_step(
            nn, batch_target_tensor, batch_input_tensor, criterion,
            optimizer=optimizer)
        train_losses.append(train_loss)

        if print_every and len(train_losses) % print_every == 0:
            logger.info(" • Iter {:1d} ({}s) | Train Batch Acc: {:.2f}, Loss: {:.5f}".format(
                len(train_losses), datetime.now() - start_time, accuracy, train_loss))

        del batch_input_tensor
        del batch_target_tensor

    mean_train_loss = mean(train_losses)
    return mean_train_loss, nn


def eval_epoch(nn, val_dl, criterion, start_time=datetime.now(), epoch_num=0):
    nn = nn.eval()
    eval_accuracies = []
    eval_losses = []
    for batch in val_dl:
        batch_input_tensor, batch_target_tensor = batch
        eval_accuracy, eval_loss = evaluate_step(
            nn, batch_target_tensor, batch_input_tensor, criterion)
        eval_accuracies.append(eval_accuracy)
        eval_losses.append(eval_loss)

        del batch_input_tensor
        del batch_target_tensor

    mean_eval_loss = mean(eval_losses)
    mean_eval_acc = mean(eval_accuracies)

    logger.info("Epoch {:1d} ({}s) | Validation Acc: {:.2f}, Loss: {:.5f}".format(
        epoch_num, datetime.now() - start_time, mean_eval_acc, mean_eval_loss))

    return mean_eval_loss, nn


def train_full(
        nn, max_window_size=None, learning_rate=0.001, patience_threshold=-1,
        n_epochs=200, batch_size=128, print_every=None, use_cuda=False):
    # Instantiate the dataset
    logger.info("Initializing the dataset...")
    dataset = CharDataset(
        # data_path="./data/debugging_charseqs.json",  # debugging
        max_window_size=max_window_size,
        transform=transforms.Compose(
            [CharSequenceToTensor(cuda=use_cuda), ]))

    logger.info("Training the neural network...")

    logger.info(" • learning rate: {}".format(learning_rate))
    logger.info(" • num epochs: {}".format(n_epochs))
    logger.info(" • batch size: {}".format(batch_size))
    logger.info(" • max window size: {}".format(max_window_size))

    train_epoch_losses = []
    eval_epoch_losses = []
    patience_counter = 0
    interrupted = False

    train_len = floor(len(dataset) * 0.9)
    val_len = len(dataset) - train_len

    logger.info("Dataset contains {} data samples".format(len(dataset)))
    if logger.getEffectiveLevel() <= logging.DEBUG:
        item_sizes = {}
        for sample in dataset:
            x_tensor, _ = sample
            window_size, batch_size, _ = x_tensor.size()
            item_sizes[window_size] = item_sizes.get(window_size, 0) + 1
        for k, v in item_sizes.items():
            logger.debug(" • {} batch_item of sequence length {}".format(v, k))
        del item_sizes

    logger.info("Dataset split {}:{}".format(train_len, val_len))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, (train_len, val_len))

    os_cpus = min(1, len(os.sched_getaffinity(0)))
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=os_cpus,
        collate_fn=batch_collate_pairs
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=os_cpus,
        collate_fn=batch_collate_pairs
    )
    logger.info(" • train on {} data samples".format(len(train_ds)))
    logger.info("   • {} training batches per epoch".format(len(train_dl)))

    logger.info(" • validate on {} data samples".format(len(val_ds)))
    logger.info("   • {} validation batches per epoch".format(len(val_dl)))

    start = datetime.now()
    criterion = NLLLoss()
    eval_loss = 0
    logger.info("Training started {}".format(start))
    try:
        for epoch in range(n_epochs):
            train_loss, nn = train_epoch(
                nn, train_dl, criterion, lr=learning_rate, start_time=start, print_every=print_every)
            train_epoch_losses.append(train_loss)
            eval_loss, nn = eval_epoch(nn, val_dl, criterion,
                                       start_time=start,
                                       epoch_num=len(eval_epoch_losses))

            if patience_threshold > 0 and eval_epoch_losses:
                if eval_loss < min(eval_epoch_losses):
                    patience_counter = 0
                else:
                    patience_counter += 1
            eval_epoch_losses.append(eval_loss)

            # only save the best models
            if eval_loss == min(eval_epoch_losses):
                nn.save(epoch=epoch, loss=eval_loss)
            if patience_counter >= patience_threshold:
                logger.info(
                    "No evaluation loss improvement in {} epochs!".format(patience_counter))
                break

    except KeyboardInterrupt:
        logger.warning("...Interrupted")
        interrupted = True
    finally:
        model_path = nn.save(epoch=epoch, loss=eval_loss,
                             interrupted=interrupted)
        logger.info("State dictionary saved to {}".format(model_path))

        progress_path = nn.save_progress({
            "net": "{}".format(nn),
            "max_num_epochs": n_epochs,
            "batch_size": batch_size,
            "max_window_size": max_window_size,
            "patience_threshold": patience_threshold,
            "train_losses": train_epoch_losses,
            "eval_losses": eval_epoch_losses})
        logger.info("Training Progress saved to {}".format(progress_path))
        logger.info("Training ended {}".format(datetime.now()))
