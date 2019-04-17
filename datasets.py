import json
import torch
import math
import numpy as np
from token import INDENT, NT_OFFSET, tok_name
from io import StringIO
from tokenize import EXACT_TOKEN_TYPES, generate_tokens
from torch.utils.data import Dataset
from torchvision import transforms


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
        for char_seq_idx, char_sequence in enumerate(self.char_sequences):
            window_size = len(char_sequence)
            if not self.max_window_size is None:
                window_size = min(self.max_window_size, window_size)
            chunk_windows = range(0, len(char_sequence) - window_size + 1)
            for start_idx in chunk_windows:
                self.data_idx_to_seq_idxs[data_idx] = (char_seq_idx, start_idx)
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


def batch_collate_pairs(samples):
    # samples is a list of (inp_seq_tensor, target_seq_tensor) pairs
    inp_seq_tensors, target_seq_tensors = zip(*samples)
    inp_tensor_batch = torch.cat(inp_seq_tensors, dim=1)
    target_tensor_batch = torch.cat(target_seq_tensors, dim=1)
    return (inp_tensor_batch, target_tensor_batch)

# d = CharDataset(transform=transforms.Compose([CharSequenceToTensor(),]))


class TokenDataset(Dataset):
    """
    Token sequences dataset.
    Same files as the Character dataset represented using Tokens.
    We will keep track of the token types and literals in separate vocabularies
    and perform backprop using two outputs.
    """
    _TOKEN_IDS_TO_TYPE = {**tok_name}
    # get rid of NT_OFFSET, we only care about terminal tokens
    del _TOKEN_IDS_TO_TYPE[NT_OFFSET]

    # these need to be squashed down such that the ids are dense
    _LITERAL_TOKENS = list(EXACT_TOKEN_TYPES.items())
    _LITERAL_TOKENS.sort()

    # token vocabulary lookup table
    INT2TOKENTYPE = {**_TOKEN_IDS_TO_TYPE}
    TOKENTYPE2INT = {t_val: t_id for t_id, t_val in INT2TOKENTYPE.items()}
    INT2LITERAL = {lit_id: lit_val[0] for lit_id, lit_val in enumerate(_LITERAL_TOKENS)}
    LITERAL2INT = {lit_val: lit_id for lit_id, lit_val in INT2LITERAL.items()}

    # no longer need these, clean them up
    del _LITERAL_TOKENS
    del _TOKEN_IDS_TO_TYPE

    def __init__(self, data_path="./data/charseqs.json", max_window_size=None, transform=None):
        super(TokenDataset, self).__init__()

        # load the raw character json file
        with open(data_path, "r") as f:
            char_sequences = json.load(f)

        # convert each file into a sequence of Python tokens
        for char_sequence in char_sequences:
            # do not count the FILE_START and FILE_END characters
            filetext = "".join(char_sequence[1:-1])
            file_tokens = self._convert_text_to_tokens(filetext)
            self._append_to_vocabulary(file_tokens)

    @classmethod
    def _append_to_vocabulary(cls, tokens):
        # append all new tokens into the class vocabulary lookup table
        for token_pair in tokens:
            t_id, t_val = token_pair
            if t_id in EXACT_TOKEN_TYPES.values():
                # this token type is exact, don't add it to the vocabulary
                continue

            # has this token/literal pair already been processed?
            tid_exist = cls.LITERAL2INT.get(t_val, None)
            if tid_exist is None:
                # otherwise add this new t_id/t_val to the vocab
                tid_new = max(cls.INT2LITERAL.keys()) + 1
                cls.LITERAL2INT[t_val] = tid_new
                cls.INT2LITERAL[tid_new] = t_val

    @staticmethod
    def _convert_text_to_tokens(text, tab="    "):
        temp_file_readline = StringIO(text).readline
        gen = generate_tokens(temp_file_readline)
        file_tokens = []
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
        return 1

    def __getitem__(self, index):
        return 1

    @classmethod
    def get_type_vocabulary(cls):
        return cls.INT2TOKENTYPE, cls.TOKENTYPE2INT

    @classmethod
    def get_literal_vocabulary(cls):
        return cls.INT2LITERAL, cls.LITERAL2INT