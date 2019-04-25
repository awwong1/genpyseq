#!/usr/bin/env python3
import json
import os
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from io import StringIO
from glob import glob
from multiprocessing import Pool
from tokenize import untokenize
from token import tok_name, NT_OFFSET

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


def check_token_sequence_is_parseable(sequence):
    try:
        _ids_to_type = {**tok_name}
        del _ids_to_type[NT_OFFSET]
        FILE_START = max(_ids_to_type.keys())
        PAD = FILE_START + 1
        # remove custom START tokens from our list of tokens
        tokens = [t for t in sequence if t[0] not in (FILE_START, PAD)]
        source_code = untokenize(tokens)
        return 1, source_code
    except:
        return 0, None


def main(data_file_glob=""):
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    file_paths = glob(data_file_glob)
    for file_path in file_paths:
        with open(file_path, "r") as f:
            sequences = json.load(f)
        executable = []
        start = datetime.now()
        for sequence in sequences:
            if all(len(elem) == 1 for elem in sequence):
                # it's a character sequence
                res = check_char_sequence_is_executable(sequence)
                logger.info("{} ({})".format(len(executable), datetime.now() - start))
                executable.append(res)
            elif all(len(elem) == 2 for elem in sequence):
                # it's a token sequence
                parseable, source_code = check_token_sequence_is_parseable(sequence)
                res = 0
                if parseable:
                    res = check_char_sequence_is_executable(source_code)
                executable.append((parseable, res))

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
