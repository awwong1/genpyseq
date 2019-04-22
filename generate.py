import os
import torch
import logging
from io import StringIO
from token import ENDMARKER
from tokenize import generate_tokens, untokenize
from datasets import CharDataset, CharSequenceToTensor, TokenDataset, TokenSequenceToTensor

logger = logging.getLogger("genpyseq")


def generate_charseq(
        nn, prime_str=CharDataset.FILE_START, max_window_size=None, print_output=True,
        max_generate_len=1000, temperature=None):
    progress_path = nn.get_progress_path()
    # Load our model
    logger.info("Loading the model weights...")
    path = nn.get_state_dict_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            ("Model does not exist at {}. " +
                "Manual model renaming required.").format(path))
    nn.load_state_dict(torch.load(path))
    nn = nn.eval()

    logger.info(" • max window size: {}".format(max_window_size))
    logger.info(" • temperature: {}".format(temperature))
    logger.info(" • max generate length: {}".format(max_generate_len))
    logger.info("Generating sequence.")

    if not prime_str.startswith(CharDataset.FILE_START):
        prime_str = CharDataset.FILE_START + prime_str

    hidden = nn.init_hidden(1)
    input_seq = list(prime_str)

    if print_output:
        print("~~~~~~~~~Prime~~~~~~~~~")
        print("".join(input_seq))
        print("~~~~Prime+Predicted~~~~")
        print("".join(input_seq))

    window_size = len(input_seq)
    if max_window_size is not None:
        window_size = max_window_size

    charseq_to_tensor = CharSequenceToTensor()

    # use priming sequence to construct the hidden state
    input_tensor, _ = charseq_to_tensor(
        (input_seq[-window_size:], input_seq[-window_size:]))
    for i in range(len(input_seq)):
        output, hidden = nn(input_tensor.narrow(0, i, 1), hidden)

    # predict until max_len or FILE_END/PAD character is reached
    predicted = input_seq[:]
    for i in range(max_generate_len):
        if temperature is not None:
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            char = CharDataset.INT2CHAR[top_i.item()]
        else:
            _, pred_char_idx = output.topk(1)
            char = CharDataset.INT2CHAR[pred_char_idx.item()]

        if print_output:
            print(char, end="")
        predicted.append(char)

        if char in [CharDataset.FILE_END, CharDataset.PAD]:
            return predicted

        # full start to end continuation of hidden state
        if max_window_size is None:
            input_tensor, _ = charseq_to_tensor(((char,), (char,)))
            output, hidden = nn(input_tensor, hidden)
        else:
            # reconstruct hidden state based on sliding window
            hidden = nn.init_hidden(1)
            input_tensor, _ = charseq_to_tensor(
                (predicted[-window_size:], predicted[-window_size:]))

            seq_len, _, _ = input_tensor.size()
            for h_i in range(seq_len):
                output, hidden = nn(input_tensor.narrow(0, h_i, 1), hidden)

    if print_output:
        print("~~max_gen_len reached~~")
    return predicted


def generate_tokenseq(
        nn, prime_str="", max_window_size=None, print_output=True,
        max_generate_len=1000, temperature=None):
    progress_path = nn.get_progress_path()
    # Load our model
    logger.info("Loading the model weights...")
    path = nn.get_state_dict_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            ("Model does not exist at {}. " +
                "Manual model renaming required.").format(path))
    nn.load_state_dict(torch.load(path))
    nn = nn.eval()

    logger.info(" • max window size: {}".format(max_window_size))
    logger.info(" • temperature: {}".format(temperature))
    logger.info(" • max generate length: {}".format(max_generate_len))
    logger.info("Generating sequence.")

    file_tokens = [(TokenDataset.FILE_START,
                    TokenDataset.INT2TOKENTYPE[TokenDataset.FILE_START]), ]

    type_hidden, literal_hidden = nn.init_hidden(1)

    gen = generate_tokens(StringIO(prime_str).readline)
    for token in gen:
        if token.type != ENDMARKER:
            file_tokens.append((token.type, token.string))

    if print_output:
        print("~~~~~~~~~Prime~~~~~~~~~")
        print(*file_tokens, sep="")
        print("~~~~Prime+Predicted~~~~")
        print(*file_tokens, sep="", end="")

    window_size = len(file_tokens)
    if max_window_size is not None:
        window_size = max_window_size
    tokenseq_to_tensor = TokenSequenceToTensor()

    # use priming sequence to construct the hidden state
    input_tensor, _ = tokenseq_to_tensor(
        (file_tokens[-window_size:], file_tokens[-window_size:]))
    for i in range(len(file_tokens)):
        inp_type_tensor, inp_literal_tensor = input_tensor
        type_output, literal_output, type_hidden, literal_hidden = nn(inp_type_tensor.narrow(
            0, i, 1), inp_literal_tensor.narrow(0, i, 1), type_hidden, literal_hidden)

    # predict until max_len or FILE_END/PAD character is reached
    predicted = file_tokens[:]
    for i in range(max_generate_len):
        if temperature is not None:
            # Sample from the network as a multinomial distribution
            type_output_dist = type_output.data.view(-1).div(temperature).exp()
            type_top_i = torch.multinomial(type_output_dist, 1)[0]
            token_type = type_top_i.item()
            literal_output_dist = literal_output.data.view(
                -1).div(temperature).exp()
            literal_top_i = torch.multinomial(literal_output_dist, 1)[0]
            token_literal = TokenDataset.INT2LITERAL[literal_top_i.item()]
        else:
            _, pred_type_idx = type_output.topk(1)
            _, pred_literal_idx = literal_output.topk(1)
            token_type = pred_type_idx.item()
            token_literal = TokenDataset.INT2LITERAL[pred_literal_idx.item()]

        if print_output:
            print((token_type, token_literal), end="")
        predicted.append((token_type, token_literal))

        if token_type in [TokenDataset.FILE_END, TokenDataset.PAD]:
            break

        # full start to end continuation of hidden state
        if max_window_size is None:
            input_tensor, _ = tokenseq_to_tensor(
                (((token_type, token_literal),), ((token_type, token_literal),)))
            inp_type_tensor, inp_literal_tensor = input_tensor
            type_output, literal_output, type_hidden, literal_hidden = nn(
                inp_type_tensor, inp_literal_tensor, type_hidden, literal_hidden)
        else:
            # reconstruct hidden state based on sliding window
            type_hidden, literal_hidden = nn.init_hidden(1)
            input_tensor, _ = tokenseq_to_tensor(
                (predicted[-window_size:], predicted[-window_size:]))

            seq_len, _, _ = input_tensor.size()
            for h_i in range(seq_len):
                inp_type_tensor, inp_literal_tensor = input_tensor
                type_output, literal_output, type_hidden, literal_hidden = nn(inp_type_tensor.narrow(
                    0, i, 1), inp_literal_tensor.narrow(0, i, 1), type_hidden, literal_hidden)

    if print_output and i == max_generate_len - 1:
        print("~~max_gen_len reached~~")
    print(untokenize(predicted))
    return predicted
