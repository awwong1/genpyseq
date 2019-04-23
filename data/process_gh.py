#!/usr/bin/env python3
"""Utility script for preprocessing the GitHub extracted Python dataset
"""
import sys
import logging
import json
import os
import math
import numpy as np
from argparse import ArgumentParser
from ast import parse
from datetime import datetime
from io import StringIO
from token import ERRORTOKEN, INDENT
from tokenize import generate_tokens, untokenize
from glob import glob
from multiprocessing import Pool

logger = logging.getLogger("genpyseq")

# possible characters to evaluate
VALID_CHAR_IDS = (9, 10, ) + tuple(range(32, 127))


def process_sample_file(gh_extract_line, tab="    "):
    try:
        gh_extract = json.loads(gh_extract_line)
        sample_filepath = gh_extract.get("sample_path", "<unknown>")
        raw_python_source = gh_extract.get("content", None)

        # Check 1. Assert file contains only valid subset of ASCII
        assert all(ord(char) in VALID_CHAR_IDS for char in raw_python_source)

        # Check 2. Ensure that the source code can be tokenized without error
        tokens = []
        _pre_norm_indents = []
        gen = generate_tokens(StringIO(raw_python_source).readline)
        for token in gen:
            assert token.type != ERRORTOKEN
            tokens.append((token.type, token.string))
            if token.type == INDENT:
                _pre_norm_indents.append(token.string)

        if _pre_norm_indents:
            indent_lens = [len(indent) for indent in _pre_norm_indents]
            # greatest common denominator of all tabs is base tabsize
            _tabsize = int(np.gcd.reduce(indent_lens))
            for ft_idx, token_pair in enumerate(tokens):
                t_type, t_val = token_pair
                if t_type == INDENT:
                    num_indents = math.floor(len(t_val) / _tabsize)
                    tokens[ft_idx] = (t_type, tab * num_indents)
        python_source = untokenize(tokens)

        # Check 3. Assert the character length is between 100 and 50000
        assert 100 <= len(python_source) <= 50000

        # Check 4. Ensure that the source code can be parsed into an AST
        node = parse(python_source, filename=sample_filepath)

        # update the dictionary with the evaluated file information
        file_id = gh_extract["id"]
        return file_id, python_source, tokens
    except:
        return None, "", []


def main(charseq_out="", tokenseq_out=""):
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    start = datetime.now()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    raw_files_path = os.path.join(dir_path, "gh_raw_extract", "*.json")
    files_to_evaluate = glob(raw_files_path)
    # evaluated_files = {}
    files_chars = []
    files_tokens = []
    num_files = 0
    num_chars = 0
    num_tokens = 0
    for file_name in files_to_evaluate:
        base_name = os.path.basename(file_name)
        with open(file_name, "r") as f:
            # with Pool() as p:
            #     it = p.imap(process_sample_file, f)
            for line in f:
                result = process_sample_file(line)
                # for result in it:
                file_id, content, tokens = result
                if file_id is not None and content and len(tokens):
                    # evaluated_files[file_id] = {
                    #     "content": content,
                    #     "tokens": tokens
                    # }
                    num_files += 1
                    num_chars += len(content)
                    num_tokens += len(tokens)
                    logger.info("{} ({}) f: {} c: {} t: {}".format(
                        base_name, datetime.now() - start, num_files, num_chars, num_tokens))
                    file_chars = (chr(2), ) + tuple(content) + (chr(3),)
                    files_chars.append(file_chars)
                    files_tokens.append(tokens)
                if num_files >= 1000:
                    break
        if num_files >= 1000:
            break

        # evaluated_path = os.path.join(dir_path, "gh_processed", base_name)
        # with open(evaluated_path, "w") as f:
        #     json.dump(evaluated_files, f)
        # evaluated_files = {}
    with open(charseq_out, "w") as f:
        json.dump(files_chars, f, separators=(',', ':'))
    with open(tokenseq_out, "w") as f:
        json.dump(files_tokens, f, separators=(",", ":"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--charseq-out", help="output of char seq file",
               type=str, default="charseqs.json")
    parser.add_argument("--tokenseq-out", help="output of token seq file",
               type=str, default="tokenseqs.json")
    args = parser.parse_args()
    main(charseq_out=args.charseq_out, tokenseq_out=args.tokenseq_out)
