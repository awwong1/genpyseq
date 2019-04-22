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

from datasets import CharDataset, TokenDataset

logger = logging.getLogger("genpyseq")


def warn_batch_window_mismatch(window_size, batch_size):
    # Warn if window_size is None, batch_size should be 1
    if window_size is None and batch_size is not 1:
        logger.warning("~" * 40)
        logger.warning(
            "WARN: Undefined window_size with batch_size: {}".format(batch_size))
        logger.warning(
            "\tBatches may not have equal sequence lengths!")
        logger.warning(
            "\tWindow size should be defined when batch_size > 1.")
        logger.warning("~" * 40)


class CharTrainer(object):
    @staticmethod
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

    @staticmethod
    def evaluate_step(nn, target_tensor, input_tensor, criterion):
        return CharTrainer.train_step(nn, target_tensor, input_tensor, criterion, eval_only=True)

    @staticmethod
    def train_epoch(
            nn, train_dl, criterion, lr=0.001,
            start_time=datetime.now(), print_every=None):
        nn = nn.train()
        optimizer = Adam(nn.parameters(), lr=lr)

        train_losses = []
        for batch in train_dl:
            batch_input_tensor, batch_target_tensor = batch
            accuracy, train_loss = CharTrainer.train_step(
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

    @staticmethod
    def eval_epoch(nn, val_dl, criterion, start_time=datetime.now(), epoch_num=0):
        nn = nn.eval()
        eval_accuracies = []
        eval_losses = []
        for batch in val_dl:
            batch_input_tensor, batch_target_tensor = batch
            eval_accuracy, eval_loss = CharTrainer.evaluate_step(
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

    @staticmethod
    def train_full(
            nn, dataset, max_window_size=None, learning_rate=0.001, patience_threshold=-1,
            n_epochs=200, batch_size=128, print_every=None, use_cuda=False):

        warn_batch_window_mismatch(max_window_size, batch_size)
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
                logger.debug(
                    " • {} batch_item of sequence length {}".format(v, k))
            del item_sizes

        logger.info("Dataset split {}:{}".format(train_len, val_len))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, (train_len, val_len))

        os_cpus = min(1, len(os.sched_getaffinity(0)))
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=os_cpus,
            collate_fn=CharDataset.batch_collate_pairs
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=True, num_workers=os_cpus,
            collate_fn=CharDataset.batch_collate_pairs
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
                train_loss, nn = CharTrainer.train_epoch(
                    nn, train_dl, criterion, lr=learning_rate, start_time=start, print_every=print_every)
                train_epoch_losses.append(train_loss)
                eval_loss, nn = CharTrainer.eval_epoch(nn, val_dl, criterion,
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


class TokenTrainer(object):
    @staticmethod
    def train_step(
            nn, target_tensors, input_tensors, criterion,
            optimizer=None, eval_only=False):
        """Perform one step of training for an input/target tensor pair.
        """
        input_type_tensor, input_literal_tensor = input_tensors
        target_type_tensor, target_literal_tensor = target_tensors
        window_size, batch_size, _ = input_type_tensor.size()
        type_hidden, literal_hidden = nn.init_hidden(batch_size)

        nn.zero_grad()
        loss = 0
        type_match = []
        literal_match = []
        for i in range(window_size):
            _i_loss = 0
            type_output, literal_output, type_hidden, literal_hidden = nn(
                input_type_tensor.narrow(0, i, 1),
                input_literal_tensor.narrow(0, i, 1),
                type_hidden, literal_hidden)
            _, target_type = target_type_tensor[i].topk(1)
            _, target_literal = target_literal_tensor[i].topk(1)
            _, predict_type = type_output.topk(1)
            _, predict_literal = literal_output.topk(1)

            for batch_item_match in target_type.view(-1) == predict_type.view(-1):
                type_match.append(batch_item_match.item())
            for batch_item_match in target_literal.view(-1) == predict_literal.view(-1):
                literal_match.append(batch_item_match.item())

            _i_loss += criterion(type_output.view(batch_size, -1),
                                 target_type.view(-1))
            _i_loss += criterion(literal_output.view(batch_size, -1),
                                 target_literal.view(-1))
            loss += _i_loss / 2

        type_accuracy = mean(type_match)
        literal_accuracy = mean(literal_match)

        if not eval_only:
            loss.backward()
            optimizer.step()

        return (type_accuracy, literal_accuracy), loss.item() / window_size

    @staticmethod
    def evaluate_step(nn, target_tensors, input_tensors, criterion):
        return TokenTrainer.train_step(nn, target_tensors, input_tensors, criterion, eval_only=True)

    @staticmethod
    def train_epoch(
            nn, train_dl, criterion, lr=0.001,
            start_time=datetime.now(), print_every=None):
        nn = nn.train()
        optimizer = Adam(nn.parameters(), lr=lr)

        train_losses = []
        for batch in train_dl:
            batch_input_tensors, batch_target_tensors = batch
            accuracy, train_loss = TokenTrainer.train_step(
                nn, batch_target_tensors, batch_input_tensors, criterion,
                optimizer=optimizer)
            train_losses.append(train_loss)

            type_accuracy, literal_accuracy = accuracy

            if print_every and len(train_losses) % print_every == 0:
                logger.info(" • Iter {:1d} ({}s) | Train Batch Type Acc: {:.2f}, Literal Acc: {:.2f}, Loss: {:.5f}".format(
                    len(train_losses), datetime.now() - start_time, type_accuracy, literal_accuracy, train_loss))

            del batch_input_tensors
            del batch_target_tensors

        mean_train_loss = mean(train_losses)
        return mean_train_loss, nn

    @staticmethod
    def eval_epoch(nn, val_dl, criterion, start_time=datetime.now(), epoch_num=0):
        nn = nn.eval()
        eval_type_accuracies = []
        eval_literal_accuracies = []
        eval_losses = []
        for batch in val_dl:
            batch_input_tensor, batch_target_tensor = batch
            eval_accuracy, eval_loss = TokenTrainer.evaluate_step(
                nn, batch_target_tensor, batch_input_tensor, criterion)
            eval_type_accuracy, eval_literal_accuracy = eval_accuracy
            eval_type_accuracies.append(eval_type_accuracy)
            eval_literal_accuracies.append(eval_literal_accuracy)
            eval_losses.append(eval_loss)

            del batch_input_tensor
            del batch_target_tensor

        mean_eval_loss = mean(eval_losses)
        mean_eval_type_acc = mean(eval_type_accuracies)
        mean_eval_literal_acc = mean(eval_literal_accuracies)

        logger.info("Epoch {:1d} ({}s) | Val Type Acc: {:.2f}, Val Literal Acc: {:.2f}, Loss: {:.5f}".format(
            epoch_num, datetime.now() - start_time, mean_eval_type_acc, mean_eval_literal_acc, mean_eval_loss))

        return mean_eval_loss, nn

    @staticmethod
    def train_full(
            nn, dataset, max_window_size=None, learning_rate=0.001, patience_threshold=-1,
            n_epochs=200, batch_size=128, print_every=None, use_cuda=False):

        warn_batch_window_mismatch(max_window_size, batch_size)
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
        logger.info("Dataset split {}:{}".format(train_len, val_len))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, (train_len, val_len))

        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=TokenDataset.batch_collate_pairs
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=True,
            collate_fn=TokenDataset.batch_collate_pairs
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
                train_loss, nn = TokenTrainer.train_epoch(
                    nn, train_dl, criterion, lr=learning_rate, start_time=start,
                    print_every=print_every)
                train_epoch_losses.append(train_loss)
                eval_loss, nn = TokenTrainer.eval_epoch(
                    nn, val_dl, criterion, start_time=start,
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
