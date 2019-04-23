#!/usr/bin/env python3
"""Given source code as input, evaluate code 'fitness' as defined in the paper.
fitness = perplexity + (10∗parse) + (100∗execute)
"""
import kenlm
from argparse import ArgumentParser


def char_seq_to_kenlm_sentence(char_sequence):
    # remove file start and stop characters
    if char_sequence[0] == chr(2):
        char_sequence = char_sequence[1:]
    if char_sequence[-1] == chr(3):
        char_sequence = char_sequence[:-1]
    for idx, char in enumerate(char_sequence):
        # rough escape all newline characters
        if char == "\n":
            char_sequence[idx] == repr(char)
    file_content = "".join(char_sequence)
    words = file_content.split()
    return " ".join(words)


def main(ngram_model=""):
    ngram = kenlm.Model(ngram_model)
    print(ngram)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ngram-model", help="NGram model file to use",
                        type=str, default="./models/py-10gram.mmap")

    args = parser.parse_args()
    main(ngram_model=args.ngram_model)
