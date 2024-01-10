## ACPs

Improving identification of anticancer peptides based on peptide residue composition and physiochemical properties information 

The peptides sequence and labels can be found in ./data_cache.
The codes are available in ./src.

### Test the model on test set

```bash
cd ./src/
python predict.py Test
```
### Test the model on a new test set
Put the sequence to be detected into the ACP20mainNew.fasta file in fasta format.
Then, run the code below, which will generate predicted probabilities.
```bash
cd ./src/
python predict.py New
```
### contact
Kang Xiao: xiaokangneuq@163.com
