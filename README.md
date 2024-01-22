# Effective identification and differential analysis of anticancer peptides

The peptides sequence and labels can be found in .fasta type files
The codes are available in predict.py

### Test the model on test set

```bash
python predict.py Test
```
### Test the model on a new test set
Put the sequence to be detected into the ACP20mainNew.fasta file in fasta format.
Then, run the code below, which will generate predicted probabilities.
```bash
python predict.py New
```

