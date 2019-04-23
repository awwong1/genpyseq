# genpyseq

Using neural networks to generate sequences of Python source code.

See [notebooks](notebooks) for proof of concepts and examples.

## Quickstart

```bash
# setting up the development environment
virtualenv venv --python=python3
source venv activate
pip install -r requirements.txt
./main.py --help

# setting up KenLM
git submodule init
git submodule update
cd kenlm
mkdir -p build
cd build
cmake .. -DKENLM_MAX_ORDER=10
make -j 4
cd ../..
./stream_dataset_source_code.py | ./kenlm/build/bin/lmplz --verbose_header --order 10 --temp_prefix /tmp/ --arpa ./models/py-10gram.arpa
./kenlm/build/bin/build_binary ./models/py-10gram.arpa ./models/py-10gram.mmap
```

## License

[MIT License](LICENSE).

```text
MIT License
Copyright (c) 2019 Alexander William Wong
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```