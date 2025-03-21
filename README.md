# ORMA: *O*ptimal T*R*ansport-based *M*ulti-grained *A*lignments

Code from "Exploring optimal transport-based multi-grained alignments for text-molecule retrieval" (IEEE BIBM 2024)

## Training Models
To train the model, use the following command:

```bash
bash train.sh ${CUDA_DEVICE}
```

## File Structure
The project consists of the following files:
- `data/`: We use the ChEBI-20 dataset from [text2mol](https://github.com/cnedwards/text2mol) for the main experiments. For training, val and test sets, we discard invalid molecules without any chemical bonds. Additionally, we add CanonicalSMILES and molecule names from PubChem for these three sets.
  - `graph_data/`: unzip `mol_graphs.zip` from [text2mol](https://github.com/cnedwards/text2mol)
  - `token_embedding_dict.npy`: from [text2mol](https://github.com/cnedwards/text2mol)
  - `training.csv`: processed by `preprocess.py` based on `training.txt` from [text2mol](https://github.com/cnedwards/text2mol)
  - `val.csv`: processed by `preprocess.py` based on `val.txt` from [text2mol](https://github.com/cnedwards/text2mol)
  - `test.csv`: processed by `preprocess.py` based on `test.txt` from [text2mol](https://github.com/cnedwards/text2mol)
  - `preprocess.py`: run `python3 preprocess.py`
- `allenai_scibert_scivocab_uncased/`: [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) path.
- `config.json`
- `train.sh`
- `main.py`
- `modeling.py`
- `dataloader.py`
- `chemutils.py`
- `utils.py`
- `requirements.txt`
