#!/usr/bin/env python3
import json
import os
from argparse import ArgumentParser
from multiprocessing import Pool


def check_char_sequence_is_executable(char_sequence):
    import subprocess
    try:
        if char_sequence[0] == chr(2):
            char_sequence = char_sequence[1:]
        if char_sequence[-1] == chr(3):
            char_sequence = char_sequence[:-1]
        source = "".join(char_sequence)
        args = ["timeout", "1s", "python3", "-c", source]
        p = subprocess.run(args, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=1, encoding="utf-8")
        # if the return code is 0, it executed ok
        if p.returncode == 0:
            return 1
    except:
        pass
    return 0


def main(data_file):
    if not os.path.isfile(data_file):
        print("No file at {}".format(data_file))
        return

    with open(data_file, "r") as f:
        char_sequences = json.load(f)
    executable = []
    with Pool() as p:
        for res in p.imap(check_char_sequence_is_executable, char_sequences):
            executable.append(res)
    with open("{}.exec_res".format(data_file), "w") as f:
        json.dump(executable, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-file", help="file containing character sequences to execute",
        type=str, default="./charseqs.json")

    args = parser.parse_args()
    main(data_file=args.data_file)
