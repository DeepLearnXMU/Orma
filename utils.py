import os
import numpy as np
import logging
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def group_padding(group, max_length, padding_value=1e-8):
    mask = [torch.ones_like(tensor, dtype=torch.bool) for tensor in group]
    group_pad_seq = pad_sequence(group, batch_first=True, padding_value=padding_value)
    mask = pad_sequence(mask, batch_first=True, padding_value=0)
    group_pad_seq = group_pad_seq.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()

    if group_pad_seq.shape[1] >= max_length:
        group_pad_seq = group_pad_seq[:, :max_length, :] 
        mask = mask[:, :max_length, :]
    else:
        padding = ((0, 0), (0, max_length-group_pad_seq.shape[1]), (0, 0))
        group_pad_seq = np.pad(group_pad_seq, padding, 'constant', constant_values=padding_value)
        mask = np.pad(mask, padding, 'constant', constant_values=0)

    group_pad_seq = torch.from_numpy(group_pad_seq)
    mask = torch.from_numpy(mask[:, :, 0])
    return group_pad_seq, mask
    

def group_node_rep(mol_out, batch_size, num_part, max_atom, max_motif, padding_value=1e-8):
    '''
        seperate atom, motif and molecule representations from the output of GNN
        mol_out : output of GNN, size [num_nodes, emb_dim]
                  nodes include atom, motif and molecule
    '''
    atom_group, motif_group, mol_group = [], [], []
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom+num_motif+1

        atom_group.append(mol_out[count: count+num_atom])
        motif_group.append(mol_out[count+num_atom: count+num_all-1])
        mol_group.append(mol_out[count+num_all-1])

        count = count + num_all

    atom_group, atom_mask = group_padding(atom_group, max_atom, padding_value)
    motif_group, motif_mask = group_padding(motif_group, max_motif, padding_value)
    mol_group = torch.stack(mol_group)

    return atom_group, motif_group, mol_group, atom_mask, motif_mask


def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        
    return logger


def compute_metrics_hits(x):
    sorted_indices = np.argsort(-x, axis=1)
    arange_column = np.arange(x.shape[0])[:, None]
    ranks = np.nonzero(sorted_indices == arange_column)[1]
    ranks = ranks + 1

    metrics = {}
    metrics['Hits1'] = float(np.sum(ranks == 1)) * 100 / len(ranks)
    metrics['Hits5'] = float(np.sum(ranks <= 5)) * 100 / len(ranks)
    metrics['Hits10'] = float(np.sum(ranks <= 10)) * 100 / len(ranks)
    metrics['MRR'] = np.mean(1.0 / ranks)
    metrics['MeanRank'] = np.mean(ranks)
    return metrics


def compute_metrics_recall(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1] + 1
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 1)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind <= 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind <= 10)) * 100 / len(ind)
    metrics["MRR"] = np.mean(1.0 / ind )
    metrics["MeanRank"] = np.mean(ind)
    return metrics