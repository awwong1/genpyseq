import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Character level dataset and transformations
# Possible Characters for neural network
VALID_UNICODE_IDS = (0, 2, 3, 9, 10) + tuple(range(32, 127))
# Special Characters
PAD = chr(0)
FILE_START = chr(2)
FILE_END = chr(3)
CHARACTERS = set(chr(id) for id in VALID_UNICODE_IDS)
INT2CHAR = dict(enumerate(CHARACTERS))
CHAR2INT = {char: idx for idx, char in INT2CHAR.items()}


class CharDataset(Dataset):
    """Character sequences dataset."""

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
        if not self.max_window_size is None:
            window_size = min(self.max_window_size, window_size)

        end_idx = start_idx + window_size
        sample_chunk = sample[start_idx:end_idx]
        inp_char_seq = sample_chunk[:-1]
        target_char_seq = sample_chunk[1:]

        if self.transform:
            return self.transform((inp_char_seq, target_char_seq,))
        else:
            return inp_char_seq, target_char_seq


class CharSequenceToTensor(object):
    """Convert text sequence in sample to Tensors."""

    def __init__(self, num_chars=len(CHARACTERS), device=torch.device("cpu")):
        super(CharSequenceToTensor, self).__init__()
        self.device = device
        self.num_chars = num_chars

    def __call__(self, sample):
        # sample is an input, target sequence pair of ASCII characters
        in_char_seq, target_char_seq = sample
        window_size = len(in_char_seq)

        input_seq_tensor = torch.zeros(
            window_size, 1, self.num_chars, device=self.device)
        target_seq_tensor = torch.zeros(
            window_size, 1, self.num_chars, device=self.device)

        for seq_idx, seq_item in enumerate(in_char_seq):
            input_seq_tensor[seq_idx][0][CHAR2INT[seq_item]] = 1
            target_item = target_char_seq[seq_idx]
            target_seq_tensor[seq_idx][0][CHAR2INT[target_item]] = 1
        return input_seq_tensor, target_seq_tensor


def batch_collate_pairs(samples):
    # samples is a list of (inp_seq_tensor, target_seq_tensor) pairs
    inp_seq_tensors, target_seq_tensors = zip(*samples)
    inp_tensor_batch = torch.cat(inp_seq_tensors, dim=1)
    target_tensor_batch = torch.cat(target_seq_tensors, dim=1)
    return (inp_tensor_batch, target_tensor_batch)

# d = CharDataset(transform=transforms.Compose([CharSequenceToTensor(),]))
