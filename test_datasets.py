import os
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CharDataset, CharSequenceToTensor, batch_collate_pairs


def test_base_char_dataset():
    """Test base character dataset logic with no window size
    """
    b = CharDataset()
    # there are 200 files in the dataset
    assert len(b) == 200

    file_sizes = []
    for inp_char_seq, target_char_seq in b:
        assert len(inp_char_seq) == len(target_char_seq)
        assert inp_char_seq[1:] == target_char_seq[:-1]
        file_sizes.append(len(inp_char_seq) + 1)
    assert len(file_sizes) == len(b)
    assert min(file_sizes) == 23
    assert max(file_sizes) == 44926


def test_window_char_dataset():
    """Test base character dataset logic with window sizes"""
    c = CharDataset(max_window_size=23)
    # with max window size of 20, there are 692043 samples
    assert len(c) == 691644

    counter = 0
    for inp_char_seq, target_char_seq in c:
        assert len(inp_char_seq) == len(target_char_seq)
        # All files have at least 2 tokens and at most 22 tokens
        assert 1 < len(inp_char_seq) < 23
        assert 1 < len(target_char_seq) < 23
        counter += 1
    assert counter == len(c)

def test_dataloader_batching():
    d = CharDataset(
        max_window_size=23,
        transform=transforms.Compose([CharSequenceToTensor(), ])
    )
    
    dl = DataLoader(d,
        batch_size=128, shuffle=False,
        num_workers=len(os.sched_getaffinity(0)),
        collate_fn=batch_collate_pairs)
    assert len(dl) == 5404
    counter = 0
    for batch in dl:
        assert counter < len(dl)
        inp_seq_batch, target_seq_batch = batch
        assert inp_seq_batch.size() == target_seq_batch.size()
        counter += 1
