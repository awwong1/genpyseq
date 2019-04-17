#!/usr/bin/env python3
import logging
import os
from math import floor
from time import time
from statistics import mean
import torch
from torch.optim import Adam
from torch.nn import NLLLoss

from datasets import batch_collate_pairs

logger = logging.getLogger("genpyseq")


def train_step(
        nn, target_tensor, input_tensor, criterion,
        optimizer=None, eval_only=False, train_iter_accs=[],
        start_time=time(), print_every=None):
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
        train_iter_accs.append(float(loss.item()))

        if print_every and len(train_iter_accs) % print_every == 0:
            logger.info(" • Iter {:1d} ({:.1f}s) | Train Batch Acc: {:.2f}, Loss: {:.5f}".format(
                len(train_iter_accs), time() - start_time, mean(batch_match), loss))

    accuracy = mean(match)

    if not eval_only:
        loss.backward()
        optimizer.step()

    return accuracy, loss.item() / window_size


def evaluate_step(nn, target_tensor, input_tensor, criterion):
    return train_step(nn, target_tensor, input_tensor, criterion, eval_only=True)


def train_epoch(
        nn, train_dl, criterion, optimizer,
        train_iter_accs=[], start_time=time(), print_every=None):
    train_losses = []
    for batch in train_dl:
        batch_input_tensor, batch_target_tensor = batch
        _, train_loss = train_step(
            nn, batch_target_tensor, batch_input_tensor, criterion,
            optimizer=optimizer,
            train_iter_accs=train_iter_accs,
            start_time=start_time, print_every=print_every)
        train_losses.append(train_loss)

        del batch_input_tensor
        del batch_target_tensor

    mean_train_loss = mean(train_losses)
    return mean_train_loss


def eval_epoch(nn, val_dl, criterion, start_time=time(), epoch_num=0):
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

    logger.info("Epoch {:1d} ({:.1f}s) | Validation Acc: {:.2f}, Loss: {:.5f}".format(
        epoch_num, time() - start_time, mean_eval_acc, mean_eval_loss))

    return mean_eval_loss


def train_full(nn, dataset, learning_rate=0.001, n_epochs=200, batch_size=128, print_every=None):
    start = time()

    train_epoch_losses = []
    eval_epoch_losses = []

    train_len = floor(len(dataset) * 0.9)
    val_len = len(dataset) - train_len

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

    criterion = NLLLoss()
    optimizer = Adam(nn.parameters(), lr=learning_rate)

    try:
        for epoch in range(n_epochs):
            train_loss = train_epoch(
                nn, train_dl, criterion, optimizer, start_time=start, print_every=print_every)
            train_epoch_losses.append(train_loss)
            eval_loss = eval_epoch(nn, val_dl, criterion,
                                   start_time=start,
                                   epoch_num=len(eval_epoch_losses))
            eval_epoch_losses.append(eval_loss)

            nn.save(epoch=epoch, loss=eval_loss)
    except KeyboardInterrupt:
        logger.warning("...Interrupted")
        nn.save(epoch=len(eval_epoch_losses), loss=0, interrupted=True)
    finally:
        nn.save_progress({
            "train_losses": train_epoch_losses,
            "eval_losses": eval_epoch_losses})
