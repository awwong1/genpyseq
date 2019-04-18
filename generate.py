import os
import torch
import logging
from datasets import CharDataset, CharSequenceToTensor

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
