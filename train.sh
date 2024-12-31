#!/bin/bash
# set -x

DEVICE=${1:-0}
COMBINE=${2:-weighted}

python main.py \
    --training_data_path data/train.csv \
    --val_data_path data/val.csv \
    --text_encoder_path allenai_scibert_scivocab_uncased \
    --config_path config.json \
    --emb_path data/token_embedding_dict.npy \
    --graph_path data/graph-data \
    --output_path output \
    --device ${DEVICE} --combine ${COMBINE} \
    --save_ckp --batch_size 32
