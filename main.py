import os
import time
import json
from tqdm import tqdm
import numpy as np
import datetime
from types import SimpleNamespace

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from modeling import Orma
from dataloader import MolDataset, get_data
from utils import get_logger, compute_metrics_hits


global logger


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, default='./data/training.csv')
    parser.add_argument('--val_data_path', type=str, default='./data/val.csv')
    parser.add_argument('--text_encoder_path', type=str, default='./allenai_scibert_scivocab_uncased')
    parser.add_argument('--output_path', type=str, default='./outputs', help='dir to output')
    parser.add_argument('--config_path', type=str, default='./config.json')
    parser.add_argument('--emb_path', type=str, default='./data/token_embedding_dict.npy')
    parser.add_argument('--graph_path', type=str, default='./data/graph-data')   

    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_ckp', action='store_true', help='save checkpoints')
    parser.add_argument('--combine', type=str, default='weighted', 
                        choices=['weighted', 'mol', 'motif', 'atom', 'molatom', 'molmotif', 'motifatom'])
    
    args = parser.parse_args()
    return args


def record_metrics(model, batch_dict):
    # calculate similarity between all text and mol batches
    sim_matrix = []

    for i in tqdm(range(len(batch_dict['sent_rep']))):
        sent_rep = batch_dict['sent_rep'][i]    # [batch_size, emb_dim]
        mtoken_rep = batch_dict['mtoken_rep'][i]
        token_rep = batch_dict['token_rep'][i]   # [batch_size, num_token, emb_dim]
        token_mask = batch_dict['token_mask'][i]

        each_row = []
        for j in range(len(batch_dict['mol_rep'])):
            atom_rep = batch_dict['atom_rep'][j]
            mol_rep = batch_dict['mol_rep'][j]
            atom_mask = batch_dict['atom_mask'][j]
            motif_rep = batch_dict['motif_rep'][j]
            motif_mask = batch_dict['motif_mask'][j]

            if atom_rep.shape[0] != token_rep.shape[0] or atom_rep.shape[0] != batch_dict['atom_rep'][0].shape[0]:
                continue

            global_logits = model.get_global_logits(sent_rep, mol_rep)  # [batch_size, batch_size]
            local_logits = model.get_local_logits(token_rep, atom_rep, token_mask, atom_mask)
            
            phrase_rep, phrase_mask = model.get_phrase_rep(mtoken_rep, token_mask, motif_rep, motif_mask)
            mid_logits = model.get_mid_logits(phrase_rep, motif_rep, phrase_mask, motif_mask)
            ij_logits = model.combine_grain(global_logits, mid_logits, local_logits)
 
            ij_logits = ij_logits.cpu().detach().numpy()
            each_row.append(ij_logits)
        
        if each_row:
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)

    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)  # [len(val_data), len(val_data)]
    logger.info('sim matrix size: {}, {}'.format(sim_matrix.shape[0], sim_matrix.shape[1]))

    t2m_metrics = compute_metrics_hits(sim_matrix)
    m2t_metrics = compute_metrics_hits(sim_matrix.T)

    logger.info('Text-to-Mol:') 
    logger.info('Hits@1: {:.4f} - R@5: {:.4f} - Hits@10: {:.4f} - MRR: {:.4f} - Mean Rank: {:.4f}'.
                format(t2m_metrics['Hits1'], t2m_metrics['Hits5'], t2m_metrics['Hits10'], t2m_metrics['MRR'], t2m_metrics['MeanRank']))
    logger.info('Mol-to-Text:')
    logger.info('Hits@1: {:.4f} - R@5: {:.4f} - Hits@10: {:.4f} - MRR: {:.4f} - Mean Rank: {:.4f}'.
                format(m2t_metrics['Hits1'], m2t_metrics['Hits5'], m2t_metrics['Hits10'], m2t_metrics['MRR'], m2t_metrics['MeanRank']))
    
    return t2m_metrics['Hits1']


def train_epoch(model, loader, optimizer, training_config, device, global_step, epoch, scheduler):
    global logger
    log_step = training_config.n_display
    num_epochs = training_config.epochs

    model.train()

    total_loss = 0.
    start_time = time.time()

    for step, batch in enumerate(tqdm(loader)):
        data_batch = get_data(batch)
        data_batch = data_batch.to(device)

        loss, *_ = model(data_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss = total_loss + loss.item()
        
        global_step = global_step + 1
        if global_step % log_step == 0:
            logger.info(
                f'Epoch: {epoch}/{num_epochs}, Step: {step + 1}/{len(loader)}, '
                f'Lr: {"-".join([str("%.8f"%group["lr"]) for group in optimizer.param_groups])}, '
                f'Loss: {float(loss)}, '
                f'Time/step: {(time.time() - start_time) / log_step}'
            )
            start_time = time.time()

    total_loss = total_loss / len(loader)
    logger.info(f'Epoch {epoch}/{num_epochs} Finished, Loss: {total_loss}')

    return total_loss, global_step


def eval_epoch(model, loader, device):
    global logger
    model.eval()

    val_total_loss = 0.

    batch_dict = {
        'sent_rep': [],
        'mtoken_rep': [],
        'token_rep': [],
        'atom_rep': [],
        'motif_rep': [],
        'mol_rep': [],
        'token_mask': [],
        'atom_mask': [],
        'motif_mask': [],
    }

    with torch.no_grad():
        for batch in tqdm(loader):
            data_batch = get_data(batch)
            data_batch = data_batch.to(device)

            loss, sent_rep, mtoken_rep, token_rep, atom_rep, motif_rep, mol_rep, \
                  token_mask, atom_mask, motif_mask = model(data_batch)
            
            for key, value in zip(batch_dict.keys(), [sent_rep, mtoken_rep, token_rep, atom_rep, \
                                                      motif_rep, mol_rep, token_mask, atom_mask, motif_mask]):
                batch_dict[key].append(value)

            val_total_loss = val_total_loss + float(loss)

        val_total_loss = val_total_loss / len(loader)
        logger.info(f'Val Loss: {float(val_total_loss)}')

        r1 = record_metrics(model, batch_dict)

    return val_total_loss, r1


def main():
    global logger
    args = get_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    model_config = SimpleNamespace(**config['model'])
    training_config = SimpleNamespace(**config['training'])
    
    cur_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = get_logger(os.path.join(args.output_path, f'train_{cur_datetime}.log'))

    dataset = MolDataset(data_path=args.training_data_path, tokenizer_path=args.text_encoder_path, 
                         max_text_length=model_config.max_text_length, emb_path=args.emb_path, graph_path=args.graph_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=training_config.num_workers, collate_fn=lambda x:x)
    
    val_dataset = MolDataset(data_path=args.val_data_path, tokenizer_path=args.text_encoder_path, 
                             max_text_length=model_config.max_text_length, emb_path=args.emb_path, graph_path=args.graph_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=training_config.num_workers, collate_fn=lambda x:x)

    model = Orma(model_config, device, args.combine, args.text_encoder_path).to(device)

    num_epochs = training_config.epochs
    optimizer = optim.Adam([
                        {'params': model.params},
                        {'params': list(model.text_encoder.parameters()), 'lr': 3e-5}
                    ], lr=training_config.lr, weight_decay=training_config.decay)

    num_warmup_steps = training_config.num_warmup_steps
    num_training_steps = num_epochs * len(loader) - num_warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_training_steps) 
        
    logger.info('***** Running training *****')
    logger.info(f'Datetime = {cur_datetime}')
    logger.info(f'Device = {device}')
    logger.info(f'Epoch nums = {num_epochs}')
    logger.info(f'Batch size = {args.batch_size}')
    logger.info(f'Warmup steps = {num_warmup_steps}')
    logger.info(f'Weight decay = {training_config.decay}')
    logger.info(f'Log step = {training_config.n_display}')
    logger.info(f'Learning rate = {training_config.lr}')
    
    output_path = os.path.join(args.output_path, 'saved_models')
    if args.save_ckp and not os.path.exists(output_path):
        os.makedirs(output_path)
    global_step= 0

    for epoch in range(1, num_epochs + 1):
        loss, global_step = train_epoch(model, loader, optimizer, training_config, device, global_step, epoch, scheduler)

        output_model_path = os.path.join(output_path, f'best_model_{cur_datetime}.pth')
        torch.save(model.state_dict(), output_model_path)

        logger.info('***** Running evaluating *****')
        logger.info(f'Batch size = {args.batch_size}')
        
        val_loss, r1 = eval_epoch(model, val_loader, device)


if __name__ == '__main__':
    main()
