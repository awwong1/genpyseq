import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tokenize import untokenize

from datasets import (batch_collate_pairs, CharDataset,
                      CharSequenceToTensor, TokenDataset, TokenSequenceToTensor)


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
        assert inp_char_seq[0] == chr(2)
        assert target_char_seq[-1] == chr(3)
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


def test_padding_dataset():
    """Test padding character dataset logic"""
    p = CharDataset(
        data_path="./data/testcase_charseqs.json",
        max_window_size=300)
    # with a max window size of 300, there should be 2 samples
    assert len(p) == 2
    for inp_char_seq, target_char_seq in p:
        assert len(inp_char_seq) == len(target_char_seq)
        assert inp_char_seq[1:] == target_char_seq[:-1]
        assert len(inp_char_seq) == 299
        assert len(target_char_seq) == 299
        assert inp_char_seq[-1] == chr(0)
        assert target_char_seq[-1] == chr(0)


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
            check_chars = list('\x02"""Predict Test"""\nimport sys\nfrom os import getcwd\n\ndef main():\n    sys.stdout.write(getcwd())\n    for i in range(0, 10):\n        print("{} : Boop".format(i), i)\n    return False\n\nif __name__ == "__main__":\n    main()\n\x03')
            assert tuple(inp_tensor.size()) == (220, 1, 100)
            _, val = inp_tensor.topk(1)
            c_idxs = val.view(-1).tolist()
            test_chars = [d.INT2CHAR[ci] for ci in c_idxs]
            assert test_chars == check_chars[:-1]
            assert c_idxs[0] == 1  # ensure consistent idx

            assert tuple(target_tensor.size()) == (220, 1, 100)
            _, val = target_tensor.topk(1)
            c_idxs = val.view(-1).tolist()
            test_chars = [d.INT2CHAR[ci] for ci in c_idxs]
            assert test_chars == check_chars[1:]
            assert c_idxs[0] == 7


# Token dataset tests
TAB_SNIPPET = '"""Predict Test"""\nimport sys\nfrom os import getcwd\n\ndef main():\n\tsys.stdout.write(getcwd())\n\tfor i in range(0, 10):\n\t\tprint("{} : Boop".format(i), i)\n\treturn False\n\nif __name__ == "__main__":\n\tmain()\n'
NORMALIZED_SNIPPET = '"""Predict Test"""\nimport sys \nfrom os import getcwd \n\ndef main ():\n    sys .stdout .write (getcwd ())\n    for i in range (0 ,10 ):\n        print ("{} : Boop".format (i ),i )\n    return False \n\nif __name__ =="__main__":\n    main ()\n'


def test_text_to_token_conversion():
    tokens = TokenDataset._convert_text_to_tokens(TAB_SNIPPET)
    assert len(tokens) == 73
    assert TokenDataset.tokens_to_text(tokens) == NORMALIZED_SNIPPET


def test_token_literal_vocabulary_generation():
    d = TokenDataset(data_path="./data/testcase_charseqs.json",
                     max_window_size=None,
                     transform=None)
    int2literal, literal2int = d.get_literal_vocabulary()
    assert int2literal[47] == '"""Predict Test"""'
    assert int2literal[74] == '"__main__"'
    assert int2literal[67] == '"{} : Boop"'
    assert len(int2literal) == 84
    for lit_id, literal in int2literal.items():
        assert literal2int[literal] == lit_id
    assert len(literal2int) == 84


def test_base_token_dataset():
    b = TokenDataset(data_path="./data/testcase_charseqs.json",
                     max_window_size=None,
                     transform=None)
    # there are 2 files in the testcase dataset
    assert len(b) == 2

    file_sizes = []
    for inp_token_seq, target_token_seq in b:
        assert len(inp_token_seq) == len(target_token_seq)
        assert inp_token_seq[1:] == target_token_seq[:-1]
        file_sizes.append(len(inp_token_seq) + 1)
        assert inp_token_seq[0][0] == b.FILE_START
        assert target_token_seq[-1][0] == b.FILE_END
    assert len(file_sizes) == len(b)
    assert min(file_sizes) == 57
    assert max(file_sizes) == 73


def test_token_tensor_conversion():
    b = TokenDataset(data_path="./data/testcase_charseqs.json",
                     max_window_size=None,
                     transform=transforms.Compose([TokenSequenceToTensor(), ]))

    assert len(b) == 2
    input_token_tensor, target_token_tensor = b[0]
    assert tuple(input_token_tensor.size()) == (72, 1, 60, 84)
    assert tuple(target_token_tensor.size()) == (72, 1, 60, 84)

    input_token_tensor, target_token_tensor = b[1]
    assert tuple(input_token_tensor.size()) == (56, 1, 60, 84)
    assert tuple(target_token_tensor.size()) == (56, 1, 60, 84)
