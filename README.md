# ORMA: *O*ptimal T*R*ansport-based *M*ulti-grained *A*lignments

## Training Models
To train the model, use the following command:

```bash
bash train.sh ${CUDA_DEVICE}
```

## File Structure
The project consists of the following files:
- `data/`: We use ChEBI-20 dataset from [text2mol](https://github.com/cnedwards/text2mol) for main experiments. For training, val and test set, we discard invalid molecules and molecules with only one atom.
- `allenai_scibert_scivocab_uncased/`: [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) path.
- `config.json`
- `train.sh`
- `main.py`
- `modeling.py`
- `dataloader.py`
- `losses.py`
- `chemutils.py`
- `utils.py`
- `requirements.txt`
