#!/usr/bin/env bash

# Setup notifications
notica() { curl --data "d:$*" "https://notica.us/?aF7rC-Ou" ; }
# open this page on any devices you want to receive the notifications on: https://notica.us/?aF7rC-Ou


# Create the directory to put the raw generated sequences
mkdir -p raw_gen_seq

# Generate 10,000 character files, 1000 per temperature step
# and generate 10,000 token files, 1000 per temperature step
for temperature in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo "python3 main.py char --generate --disable-cuda --max-generate-len 50000 --temperature $temperature --generator-output-file raw_gen_seq/char_temperature$temperature.json"
    python3 main.py char --generate --disable-cuda --max-generate-len 50000 --temperature $temperature --generator-output-file raw_gen_seq/char_temperature$temperature.json
    notica "DONE char $temperature"

    echo "python3 main.py token --generate --disable-cuda --hidden-size 128 --max-generate-len 20000 --temperature $temperature --generator-output-file raw_gen_seq/token_temperature$temperature.json"
    python3 main.py token --generate --disable-cuda --hidden-size 128 --max-generate-len 20000 --temperature $temperature --generator-output-file raw_gen_seq/token_temperature$temperature.json
    notica "DONE token $temperature"
done
