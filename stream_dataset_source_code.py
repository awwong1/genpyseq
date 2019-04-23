#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from fitness import char_seq_to_kenlm_sentence


def main(data_file=""):
    """Quick conversion of source code into kenlm trainable input
    """
    with open(data_file, "r") as f:
        char_sequences = json.load(f)
    for char_sequence in char_sequences:
        print(char_seq_to_kenlm_sentence(char_sequence))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-file", help="file containing all source code character sequences",
        type=str, default="./data/charseqs.json")

    args = parser.parse_args()
    main(data_file=args.data_file)
