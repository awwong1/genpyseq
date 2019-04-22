import logging
import json
import torch
import math
import numpy as np
from datetime import datetime
from token import ENDMARKER, INDENT, NT_OFFSET, tok_name
from io import StringIO
from tokenize import EXACT_TOKEN_TYPES, generate_tokens, untokenize
from torch.utils.data import Dataset
from torchvision import transforms
from multiprocessing import Pool


_max_window_size = None
logger = logging.getLogger("genpyseq")


class CharDataset(Dataset):
    """
    Character sequences dataset. Extension of the Karpathy Char RNN blog post.
    """
    # Character level dataset and transformations
    # Possible Characters for neural network
    VALID_UNICODE_IDS = (0, 2, 3, 9, 10) + tuple(range(32, 127))
    # Special Characters
    PAD = chr(0)
    FILE_START = chr(2)
    FILE_END = chr(3)
    CHARACTERS = tuple(chr(id) for id in VALID_UNICODE_IDS)
    INT2CHAR = dict(enumerate(CHARACTERS))
    CHAR2INT = {char: idx for idx, char in INT2CHAR.items()}

    def __init__(self, data_path="./data/charseqs.json", max_window_size=None, transform=None):
        super(CharDataset, self).__init__()
        with open(data_path, "r") as f:
            self.char_sequences = json.load(f)

        self.transform = transform
        self.max_window_size = max_window_size

        # construct the window_offset to sequence mapping
        self.data_idx_to_seq_idxs = {}
        data_idx = 0
        start = datetime.now()
        with Pool(initializer=CharDataset.initialize_worker, initargs=(self.max_window_size,)) as p:
            for window_result in p.imap(CharDataset.construct_sliding_indicies, enumerate(self.char_sequences)):
                char_seq_idx, chunk_windows = window_result
                logger.info("File {} ({})".format(
                    char_seq_idx, datetime.now() - start))
                for start_idx in chunk_windows:
                    self.data_idx_to_seq_idxs[data_idx] = (
                        char_seq_idx, start_idx)
                    data_idx += 1

    def __len__(self):
        return len(self.data_idx_to_seq_idxs.keys())

    def __getitem__(self, idx):
        if not type(idx) is int:
            idx = idx.item()
        if idx >= len(self):
            raise StopIteration()
        (seq_idx, start_idx) = self.data_idx_to_seq_idxs[idx]
        sample = self.char_sequences[seq_idx]
        window_size = len(sample)
        if not (self.max_window_size is None):
            window_size = min(self.max_window_size, window_size)

        end_idx = start_idx + window_size
        chunk = sample[start_idx:end_idx]

        if self.max_window_size is not None and len(chunk) < self.max_window_size:
            # right-pad the chunk with PAD characters
            num_pad = self.max_window_size - len(chunk)
            chunk = chunk + [self.PAD, ] * num_pad

        inp_char_seq = chunk[:-1]
        target_char_seq = chunk[1:]

        if self.transform:
            return self.transform((inp_char_seq, target_char_seq,))
        else:
            return inp_char_seq, target_char_seq

    @staticmethod
    def batch_collate_pairs(samples):
        # samples is a list of (inp_seq_tensor, target_seq_tensor) pairs
        inp_seq_tensors, target_seq_tensors = zip(*samples)
        inp_tensor_batch = torch.cat(inp_seq_tensors, dim=1)
        target_tensor_batch = torch.cat(target_seq_tensors, dim=1)
        return (inp_tensor_batch, target_tensor_batch)

    @staticmethod
    def initialize_worker(max_window_size):
        global _max_window_size
        _max_window_size = max_window_size

    @staticmethod
    def construct_sliding_indicies(idx_sequence_pair):
        char_seq_idx, char_sequence = idx_sequence_pair
        if char_sequence[0] != CharDataset.FILE_START:
            char_sequence.insert(0, CharDataset.FILE_START)
        if char_sequence[-1] != CharDataset.FILE_END:
            char_sequence.append(CharDataset.FILE_END)
        window_size = len(char_sequence)
        if _max_window_size is not None:
            window_size = min(_max_window_size, window_size)
        chunk_windows = range(0, len(char_sequence) - window_size + 1)
        return char_seq_idx, chunk_windows


class CharSequenceToTensor(object):
    """Convert text sequence in sample to Tensors."""

    def __init__(self, num_chars=len(CharDataset.CHARACTERS), cuda=False):
        super(CharSequenceToTensor, self).__init__()
        self.cuda = cuda
        self.num_chars = num_chars

    def __call__(self, sample):
        # sample is an input, target sequence pair of ASCII characters
        in_char_seq, target_char_seq = sample
        window_size = len(in_char_seq)

        input_seq_tensor = torch.zeros(window_size, 1, self.num_chars)
        target_seq_tensor = torch.zeros(window_size, 1, self.num_chars)

        for seq_idx, seq_item in enumerate(in_char_seq):
            input_seq_tensor[seq_idx][0][CharDataset.CHAR2INT[seq_item]] = 1
            target_item = target_char_seq[seq_idx]
            target_seq_tensor[seq_idx][0][CharDataset.CHAR2INT[target_item]] = 1
        if self.cuda:
            return input_seq_tensor.cuda(), target_seq_tensor.cuda()
        return input_seq_tensor, target_seq_tensor

# d = CharDataset(transform=transforms.Compose([CharSequenceToTensor(),]))


class TokenDataset(Dataset):
    """
    Token sequences dataset.
    Same files as the Character dataset represented using Tokens.
    We will keep track of the token types and literals in separate vocabularies
    and perform backprop using two the outputs.
    """
    _TOKEN_IDS_TO_TYPE = {**tok_name}
    # get rid of NT_OFFSET, we only care about terminal tokens
    del _TOKEN_IDS_TO_TYPE[NT_OFFSET]
    # we need to have a start token for seeding our neural network
    FILE_START = max(_TOKEN_IDS_TO_TYPE.keys())
    FILE_END = ENDMARKER
    PAD = FILE_START + 1
    _TOKEN_IDS_TO_TYPE[FILE_START] = "STARTMARKER"
    _TOKEN_IDS_TO_TYPE[FILE_END] = "ENDMARKER"
    _TOKEN_IDS_TO_TYPE[PAD] = "PAD"

    # these need to be squashed down such that the ids are dense, 0 meaning not literal
    _LITERAL_TOKENS = list(EXACT_TOKEN_TYPES.keys()) + [chr(0), ]
    _LITERAL_TOKENS.sort()

    # token vocabulary lookup table
    INT2TOKENTYPE = {**_TOKEN_IDS_TO_TYPE}
    TOKENTYPE2INT = {t_val: t_id for t_id, t_val in INT2TOKENTYPE.items()}

    # literal vocabulary lookup table
    INT2LITERAL = {lit_id: lit_val for lit_id,
                   lit_val in enumerate(_LITERAL_TOKENS)}
    LITERAL2INT = {lit_val: lit_id for lit_id, lit_val in INT2LITERAL.items()}

    EXACT_TOKEN_TYPE_IDS = set(EXACT_TOKEN_TYPES.values())
    EXACT_TOKEN_TYPE_IDS.update([FILE_START, FILE_END, PAD])
    NON_LITERAL_STUB_ID = LITERAL2INT[chr(0)]

    # no longer need these, clean them up
    del _LITERAL_TOKENS
    del _TOKEN_IDS_TO_TYPE

    def __init__(self, data_path="./data/tokenseqs.json", max_window_size=None, transform=None):
        super(TokenDataset, self).__init__()

        self.max_window_size = max_window_size
        self.transform = transform
        self.token_sequences = []

        start = datetime.now()

        # load the raw json file
        logger.info("loading data file")
        with open(data_path, "r") as f:
            sequences = json.load(f)
        if len(sequences[0][0]) == 2:
            # these are token sequnces
            self.token_sequences = sequences
            logger.info("likely tokens")
        else:
            # these are character sequences
            logger.info("likely characters")
            with Pool() as p:
                for tokens in p.imap(TokenDataset._convert_text_to_tokens, sequences):
                    self.token_sequences.append(tokens)
                    logger.info("File {} ({})".format(
                        len(self.token_sequences), datetime.now() - start))

        # convert each file into a sequence of Python tokens
        # construct the window_offset to sequence mapping
        self.data_idx_to_seq_idxs = {}
        data_idx = 0

        for seq_idx, token_sequence in enumerate(self.token_sequences):
            if token_sequence[0][0] != self.FILE_START:
                token_sequence = [(self.FILE_START, self.INT2TOKENTYPE[self.FILE_START]),] + token_sequence
                self.token_sequences[seq_idx] = token_sequence
            self._append_to_vocabulary(token_sequence)
            logger.info("File {} ({})".format(seq_idx, datetime.now() - start))
            window_size = len(token_sequence)
            if self.max_window_size is not None:
                window_size = min(self.max_window_size, window_size)
            chunk_windows = range(0, len(token_sequence) - window_size + 1)
            for start_idx in chunk_windows:
                self.data_idx_to_seq_idxs[data_idx] = (seq_idx, start_idx)
                data_idx += 1


    @classmethod
    def _append_to_vocabulary(cls, tokens):
        # append all new tokens into the class vocabulary lookup table
        for token_pair in tokens:
            t_id, t_val = token_pair
            if t_id in (*EXACT_TOKEN_TYPES.values(), cls.FILE_START, cls.FILE_END, cls.PAD):
                # this token type is exact, don't add it to the vocabulary
                continue

            # has this token/literal pair already been processed?
            tid_exist = cls.LITERAL2INT.get(t_val, None)
            if tid_exist is None:
                # otherwise add this new t_id/t_val to the vocab
                tid_new = max(cls.INT2LITERAL.keys()) + 1
                cls.LITERAL2INT[t_val] = tid_new
                cls.INT2LITERAL[tid_new] = t_val

    @classmethod
    def _convert_text_to_tokens(cls, char_sequence, tab="    "):
        if char_sequence[0] == CharDataset.FILE_START:
            char_sequence = char_sequence[1:]
        if char_sequence[-1] == CharDataset.FILE_END:
            char_sequence.pop()
        text = "".join(char_sequence)

        temp_file_readline = StringIO(text).readline
        gen = generate_tokens(temp_file_readline)
        file_tokens = [(cls.FILE_START, cls.INT2TOKENTYPE[cls.FILE_START]), ]
        _pre_norm_indents = []
        for token in gen:
            # store all the indents for further processing
            if token.type == INDENT:
                _pre_norm_indents.append(token.string)
            file_tokens.append((token.type, token.string))
        if _pre_norm_indents:
            indent_lens = [len(x) for x in _pre_norm_indents]
            # greatest common denominator of all tabs is base tabsize
            _tabsize = int(np.gcd.reduce(indent_lens))
            for ft_idx, token_pair in enumerate(file_tokens):
                t_type, t_val = token_pair
                if t_type == INDENT:
                    num_indents = math.floor(len(t_val) / _tabsize)
                    file_tokens[ft_idx] = (t_type, tab * num_indents)
        return file_tokens

    def __len__(self):
        return len(self.data_idx_to_seq_idxs.keys())

    def __getitem__(self, idx):
        if not type(idx) is int:
            idx = idx.item()
        if idx >= len(self):
            raise StopIteration()
        (seq_idx, start_idx) = self.data_idx_to_seq_idxs[idx]
        sample = self.token_sequences[seq_idx]
        window_size = len(sample)
        if not (self.max_window_size is None):
            window_size = min(self.max_window_size, window_size)

        end_idx = start_idx + window_size
        chunk = sample[start_idx:end_idx]

        if self.max_window_size is not None and len(chunk) < self.max_window_size:
            # right-pad the chunk with PAD tokens
            num_pad = self.max_window_size - len(chunk)
            chunk = chunk + \
                [(self.PAD, self.INT2TOKENTYPE[self.PAD])] * num_pad

        inp_token_seq = chunk[:-1]
        target_token_seq = chunk[1:]

        if self.transform:
            return self.transform((inp_token_seq, target_token_seq,))
        else:
            return inp_token_seq, target_token_seq

    @classmethod
    def tokens_to_text(cls, tokens):
        """Helper method for converting sequences of tokens back to source code
        """
        # we need to remove the custom START token from our list of tokens
        _tokens = [t for t in tokens if t[0] not in (cls.FILE_START, cls.PAD)]
        return untokenize(_tokens)

    @classmethod
    def get_type_vocabulary(cls):
        return cls.INT2TOKENTYPE, cls.TOKENTYPE2INT

    @classmethod
    def get_literal_vocabulary(cls):
        return cls.INT2LITERAL, cls.LITERAL2INT

    @staticmethod
    def batch_collate_pairs(samples):
        # samples is a list of (inp_seq_tensors, target_seq_tensors) pairs
        inp_seq_tensors, target_seq_tensors = zip(*samples)
        inp_type_tensors, inp_literal_tensors = zip(*inp_seq_tensors)
        inp_type_batch_tensors = torch.cat(inp_type_tensors, dim=1)
        inp_literal_batch_tensors = torch.cat(inp_literal_tensors, dim=1)
        target_type_tensors, target_literal_tensors = zip(*target_seq_tensors)
        target_type_batch_tensors = torch.cat(target_type_tensors, dim=1)
        target_literal_batch_tensors = torch.cat(target_literal_tensors, dim=1)
        return ((inp_type_batch_tensors, inp_literal_batch_tensors),
                (target_type_batch_tensors, target_literal_batch_tensors))


class TokenSequenceToTensor(object):
    """Convert token sequence in sample to Tensors."""

    def __init__(self, cuda=False):
        super(TokenSequenceToTensor, self).__init__()
        self.cuda = cuda

    def __call__(self, sample):
        num_types = len(TokenDataset.INT2TOKENTYPE)
        num_literals = len(TokenDataset.INT2LITERAL)
        # sample is an input, target sequence pair of (TYPE_ID, LITERAL_STR)
        in_token_seq, target_token_seq = sample
        window_size = len(in_token_seq)

        input_types_tensor = torch.zeros(window_size, 1, num_types)
        input_literals_tensor = torch.zeros(window_size, 1, num_literals)
        target_types_tensor = torch.zeros(window_size, 1, num_types)
        target_literals_tensor = torch.zeros(window_size, 1, num_literals)

        for seq_idx in range(window_size):
            input_token_type_id, input_token_literal_val = in_token_seq[seq_idx]
            if input_token_type_id not in TokenDataset.EXACT_TOKEN_TYPE_IDS:
                input_token_literal_id = TokenDataset.LITERAL2INT[input_token_literal_val]
            else:
                input_token_literal_id = TokenDataset.NON_LITERAL_STUB_ID
            input_types_tensor[seq_idx][0][input_token_type_id] = 1
            input_literals_tensor[seq_idx][0][input_token_literal_id] = 1

            target_token_type_id, target_token_literal_val = target_token_seq[seq_idx]
            if target_token_type_id not in TokenDataset.EXACT_TOKEN_TYPE_IDS:
                target_token_literal_id = TokenDataset.LITERAL2INT[target_token_literal_val]
            else:
                target_token_literal_id = TokenDataset.NON_LITERAL_STUB_ID
            target_types_tensor[seq_idx][0][target_token_type_id] = 1
            target_literals_tensor[seq_idx][0][target_token_literal_id] = 1

        if self.cuda:
            return ((input_types_tensor.cuda(), input_literals_tensor.cuda()),
                    (target_types_tensor.cuda(), target_literals_tensor.cuda()))
        return ((input_types_tensor, input_literals_tensor,),
                (target_types_tensor, target_literals_tensor))
