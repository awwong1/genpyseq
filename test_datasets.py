import os
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import INT2CHAR, CharDataset, CharSequenceToTensor, batch_collate_pairs


def test_base_char_dataset():
    """Test base character dataset logic with no window size
    """
    b = CharDataset(
        data_path="./data/testcase_charseqs.json",
        max_window_size=None)
    # there are 2 files in the testcase dataset
    assert len(b) == 2

    file_sizes = []
    for inp_char_seq, target_char_seq in b:
        assert len(inp_char_seq) == len(target_char_seq)
        assert inp_char_seq[1:] == target_char_seq[:-1]
        file_sizes.append(len(inp_char_seq) + 1)
    assert len(file_sizes) == len(b)
    assert min(file_sizes) == 165
    assert max(file_sizes) == 221


def test_window_char_dataset():
    """Test base character dataset logic with window sizes"""
    c = CharDataset(
        data_path="./data/testcase_charseqs.json",
        max_window_size=20)
    # with max window size of 20, there are 348 samples
    assert len(c) == 348

    counter = 0
    for inp_char_seq, target_char_seq in c:
        assert len(inp_char_seq) == len(target_char_seq)
        # All files have at least 2 tokens and at most 22 tokens
        assert 1 < len(inp_char_seq) < 20
        assert 1 < len(target_char_seq) < 20
        counter += 1
    assert counter == len(c)


def test_dataloader_batching():
    d = CharDataset(
        data_path="./data/testcase_charseqs.json",
        max_window_size=20,
        transform=transforms.Compose([CharSequenceToTensor(), ]))
    dl = DataLoader(
        d, batch_size=128, shuffle=False,
        num_workers=len(os.sched_getaffinity(0)),
        collate_fn=batch_collate_pairs)

    # With batch size of 128, there are 3 batches (128, 128, 92)
    assert len(dl) == 3
    counter = 0
    for batch in dl:
        assert counter < len(dl)
        inp_seq_batch, target_seq_batch = batch
        assert inp_seq_batch.size() == target_seq_batch.size()
        tensor_size = tuple(inp_seq_batch.size())
        if counter == 0 or counter == 1:
            check_size = (19, 128, 100)
        elif counter == 2:
            check_size = (19, 92, 100)
        assert tensor_size == check_size
        counter += 1


def test_dataloader_vocabulary():
    d = CharDataset(
        data_path="./data/testcase_charseqs.json",
        max_window_size=None,
        transform=transforms.Compose([CharSequenceToTensor(), ]))
    dl = DataLoader(
        d, batch_size=1, shuffle=False,
        num_workers=len(os.sched_getaffinity(0)),
        collate_fn=batch_collate_pairs)

    # with batch size of 1, window size of None, two batches (whole files)
    assert len(dl) == 2
    for idx, batch in enumerate(dl):
        inp_tensor, target_tensor = batch
        if idx == 0:
            check_chars = [
                "\u0002", "\"", "\"", "\"", "P", "r", "e", "d", "i", "c", "t", " ", "T", "e", "s", "t", "\"", "\"", "\"", "\n", "i", "m", "p", "o", "r", "t", " ", "s", "y", "s", "\n", "f", "r", "o", "m", " ", "o", "s", " ", "i", "m", "p", "o", "r", "t", " ", "g", "e", "t", "c", "w", "d", "\n", "\n", "d", "e", "f", " ", "m", "a", "i", "n", "(", ")", ":", "\n", " ", " ", " ", " ", "s", "y", "s", ".", "s", "t", "d", "o", "u", "t", ".", "w", "r", "i", "t", "e", "(", "g", "e", "t", "c", "w", "d", "(", ")", ")", "\n", " ", " ", " ", " ", "f", "o", "r", " ", "i", " ", "i", "n", " ", "r", "a", "n", "g", "e", "(", "0", ",", " ", "1", "0", ")", ":", "\n", " ", " ", " ", " ", " ", " ", " ", " ", "p", "r", "i", "n", "t", "(", "\"", "{", "}", " ", ":", " ", "B", "o", "o", "p", "\"", ".", "f", "o", "r", "m", "a", "t", "(", "i", ")", ",", " ", "i", ")", "\n", " ", " ", " ", " ", "r", "e", "t", "u", "r", "n", " ", "F", "a", "l", "s", "e", "\n", "\n", "i", "f", " ", "_", "_", "n", "a", "m", "e", "_", "_", " ", "=", "=", " ", "\"", "_", "_", "m", "a", "i", "n", "_", "_", "\"", ":", "\n", " ", " ", " ", " ", "m", "a", "i", "n", "(", ")", "\n", "\u0003"]
            assert tuple(inp_tensor.size()) == (220, 1, 100)
            _, val = inp_tensor.topk(1)
            c_idxs = val.view(-1).tolist()
            test_chars = [INT2CHAR[ci] for ci in c_idxs]
            assert test_chars == check_chars[:-1]
            assert c_idxs[0] == 1 # ensure consistent idx

            assert tuple(target_tensor.size()) == (220, 1, 100)
            _, val = target_tensor.topk(1)
            c_idxs = val.view(-1).tolist()
            test_chars = [INT2CHAR[ci] for ci in c_idxs]
            assert test_chars == check_chars[1:]
            assert c_idxs[0] == 7