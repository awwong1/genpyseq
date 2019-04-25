#!/usr/bin/env python3
import json
import os
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from multiprocessing import Pool

logger = logging.getLogger(__name__)


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


def main(data_file_glob=""):
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    file_paths = glob(data_file_glob)
    for file_path in file_paths:
        with open(file_path, "r") as f:
            char_sequences = json.load(f)
        executable = []
        start = datetime.now()
        for char_sequence in char_sequences:
            res = check_char_sequence_is_executable(char_sequence)
            logger.info("{} ({})".format(len(executable), datetime.now() - start))
            executable.append(res)
        with open("{}.exec_res".format(file_path), "w") as f:
            json.dump(executable, f)
            logger.info("Saved to {}".format(f.name))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--files", help="glob containing character sequences to execute",
        type=str, default="*.json")

    args = parser.parse_args()
    main(data_file_glob=args.files)
