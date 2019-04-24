#!/usr/bin/env python3
"""Given source code as input, evaluate code 'fitness' as defined in the paper.
"""
import kenlm
import json
import subprocess
from ast import parse
from argparse import ArgumentParser
from io import StringIO
from scipy import stats
from tokenize import generate_tokens


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


def main(model_name="", data_file="", ngram_model="", exec_results="", output_name=""):
    ngram = kenlm.Model(ngram_model)

    with open(data_file, "r") as f:
        char_sequences = json.load(f)
    with open(exec_results, "r") as f:
        seq_exec_results = json.load(f)

    data_dict = {
        "Model": [],
        "Fitness": [],
        "Perplexity": [],
        "Length": [],
        "Parseability": [],
        "Executability": [],
    }

    for idx, char_sequence in enumerate(char_sequences):
        # Calculate perplexity
        kenlm_sequence = char_seq_to_kenlm_sentence(char_sequence)
        perplexity = ngram.perplexity(kenlm_sequence)

        # Calculate length
        length = char_seq_to_token_len(char_sequence)

        # Calculate parseability
        parseable = 0
        if length:
            parseable = check_char_sequence_is_parseable(char_sequence)

        # Calculate executable
        executable = 0
        if parseable:
            executable = seq_exec_results[idx]

        fitness = calculate_fitness(perplexity, length, parseable, executable)
        print("{}) PP: {:.4f}, Length: {}, Parse: {}, Exec: {} | FITNESS: {:.6f}".format(
            idx, perplexity, length, parseable, executable, fitness))
        data_dict["Model"].append(model_name)
        data_dict["Fitness"].append(fitness)
        data_dict["Perplexity"].append(perplexity)
        data_dict["Length"].append(length)
        data_dict["Parseability"].append(parseable)
        data_dict["Executability"].append(executable)

    print(stats.describe(data_dict["Fitness"]))

    with open(output_name, "w") as f:
        json.dump(data_dict, f)
        print("Saved to {}".format(f.name))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", help="model name for the dataframe",
                        type=str, default="GitHub")
    parser.add_argument("--data-file", help="file containing char sequences to evaluate",
                        type=str, default="./data/charseqs.json")
    parser.add_argument("--ngram-model", help="n-gram model file to use",
                        type=str, default="./models/py-10gram.mmap")
    parser.add_argument("--exec-results", help="file containing execution results",
                        type=str, default="./data/charseqs.json.exec_res")
    parser.add_argument("--output-name", help="name of output dataframe dict",
                        type=str, default="fitness.json")
    args = parser.parse_args()
    main(
        model_name=args.model_name, data_file=args.data_file, ngram_model=args.ngram_model,
        exec_results=args.exec_results, output_name=args.output_name)
