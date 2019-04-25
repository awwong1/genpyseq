#!/usr/bin/env python3
"""Given source code as input, evaluate code 'fitness' as defined in the paper.
"""
import re
import os
import kenlm
import json
import subprocess
from ast import parse
from argparse import ArgumentParser
from io import StringIO
from scipy import stats
from tokenize import generate_tokens
from glob import glob
from datasets import TokenDataset

def remove_start_end_chr(char_sequence):
    if char_sequence[0] == chr(2):
        char_sequence = char_sequence[1:]
    if char_sequence[-1] == chr(3):
        char_sequence = char_sequence[:-1]
    return char_sequence


def char_seq_to_kenlm_sentence(char_sequence):
    # remove file start and stop characters
    _sequence = remove_start_end_chr(char_sequence)
    for idx, char in enumerate(_sequence):
        # rough escape all newline characters
        if char == "\n":
            _sequence[idx] == repr(char)
    file_content = "".join(_sequence)
    words = file_content.split()
    return " ".join(words)


def char_seq_to_token_len(char_sequence):
    try:
        char_sequence = remove_start_end_chr(char_sequence)
        text = "".join(char_sequence)
        gen = generate_tokens(StringIO(text).readline)
        counter = 0
        for _ in gen:
            counter += 1
        return counter
    except:
        return 0


def check_char_sequence_is_parseable(char_sequence):
    try:
        char_sequence = remove_start_end_chr(char_sequence)
        source = "".join(char_sequence)
        node = parse(source)
        return 1
    except:
        return 0


def calculate_fitness(perplexity, length, parseable, executable):
    return (length + (length * parseable) + (2 * length * executable)) / perplexity


def main(data_file_glob="", ngram_model="", output_name=""):
    ngram = kenlm.Model(ngram_model)

    data_dict = {
        "Model": [],
        "Temperature": [],
        "Fitness": [],
        "Perplexity": [],
        "Length": [],
        "Parseability": [],
        "Executability": [],
    }

    data_files = glob(data_file_glob)
    for data_file in data_files:

        with open(data_file, "r") as f:
            file_name = f.name
            sequences = json.load(f)
        exec_results = "{}.exec_res".format(file_name)
        with open(exec_results, "r") as f:
            seq_exec_results = json.load(f)

        # determine the model
        m = re.search(
            "(?P<model>[a-zA-Z]*)\_temperature(?P<temperature>\d\.\d)\.json$", file_name)
        if m is None:
            model_name = "GitHub"
            temperature = 1
        else:
            params = m.groupdict()
            model_name = "{}Gen".format(params.get("model").capitalize())
            temperature = float(params.get("temperature"))

        for idx, sequence in enumerate(sequences):
            # determine if char sequence or token sequence
            if all(len(elem) == 1 for elem in sequence):
                # it's a char sequence
                # Calculate perplexity
                kenlm_sequence = char_seq_to_kenlm_sentence(sequence)
                perplexity = ngram.perplexity(kenlm_sequence)
                # Calculate length
                length = char_seq_to_token_len(sequence)
                # Calculate parseability
                parseable = 0
                if length:
                    parseable = check_char_sequence_is_parseable(sequence)
                # Calculate executable
                executable = 0
                if parseable:
                    executable = seq_exec_results[idx]
            elif all(len(elem) == 2 for elem in sequence):
                # it's a token sequence
                parseable, executable = seq_exec_results[idx]
                length = len(sequence)
                # perplexity calculation depends on parseability
                if parseable:
                    char_sequence = list(TokenDataset.tokens_to_text(sequence))
                else:
                    char_sequence = list(" ".join([token[1] for token in sequence]))
                kenlm_sequence = char_seq_to_kenlm_sentence(char_sequence)
                perplexity = ngram.perplexity(kenlm_sequence)

            fitness = calculate_fitness(
                perplexity, length, parseable, executable)
            print("{}_temp{} {}) PP: {:.4f}, Length: {}, Parse: {}, Exec: {} | FITNESS: {:.6f}".format(
                model_name, temperature, idx, perplexity, length, parseable, executable, fitness))
            data_dict["Model"].append(model_name)
            data_dict["Temperature"].append(temperature)
            data_dict["Fitness"].append(fitness)
            data_dict["Perplexity"].append(perplexity)
            data_dict["Length"].append(length)
            data_dict["Parseability"].append(parseable)
            data_dict["Executability"].append(executable)

    with open(output_name, "w") as f:
        json.dump(data_dict, f)
        print("Saved to {}".format(f.name))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--files", help="files containing char sequences to evaluate (default: ./raw_gen_seq/*.json)",
                        type=str, default="./raw_gen_seq/*.json")
    parser.add_argument("--ngram-model", help="n-gram model file to use",
                        type=str, default="./models/py-10gram.mmap")
    parser.add_argument("--output-name", help="name of output dataframe dict",
                        type=str, default="fitness.json")
    args = parser.parse_args()
    main(data_file_glob=args.files, ngram_model=args.ngram_model,
         output_name=args.output_name)
