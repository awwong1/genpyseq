Copy the sequences that you want to evaluate into this directory, then run Docker

```bash
cp ../data/charseqs.json .
docker build -t eval_untrusted .
docker run eval_untrusted
```
