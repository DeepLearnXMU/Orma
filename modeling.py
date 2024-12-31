from transformers import BertModel

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import group_node_rep

import logging
logger = logging.getLogger(__name__)


def cross_entropy_loss(logits):
    loss_func = nn.CrossEntropyLoss()
    labels = torch.arange(logits.shape[0], device=logits.device)

    sim_loss1 = loss_func(logits, labels)
    sim_loss2 = loss_func(logits.T, labels)

    return (sim_loss1 + sim_loss2) / 2


class GCNModel(nn.Module):
    def __init__(self, num_layer, emb_dim, dropout_ratio=0.5):
        super(GCNModel, self).__init__()
        self.relu = nn.ReLU()
        self.num_layer = num_layer
        self.dropout = nn.Dropout(dropout_ratio)

        #For GCN:
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(300, emb_dim))
        for _ in range(1, self.num_layer):
            self.conv_layers.append(GCNConv(emb_dim, emb_dim))

        self.mol_hidden1 = nn.Linear(emb_dim, emb_dim)
        self.mol_hidden2 = nn.Linear(emb_dim, emb_dim)
        self.mol_hidden3 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index):
        for i in range(self.num_layer):
            x = self.conv_layers[i](x, edge_index)
            x = self.dropout(x)
            if i != self.num_layer - 1: 
                x = self.relu(x)

        x = self.mol_hidden1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.mol_hidden2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.mol_hidden3(x)

        return x


class Orma(nn.Module):
    def __init__(self, config, device, combine, text_encoder_path):
        super(Orma, self).__init__()
        global logger
        
        self.num_token = config.max_text_length
        self.emb_dim = config.emb_dim
        self.hidden_size = config.hidden_size
        self.num_atom = config.num_atom
        self.num_motif = config.num_motif
        self.epsilon = config.epsilon
        self.dropout_ratio = config.dropout_ratio

        logger.info(f'num atom = {self.num_atom}, num motif = {self.num_motif}')
        logger.info(f'dropout ratio = {self.dropout_ratio}')

        self.device = device
        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.combine = combine

        self.weights = Parameter(torch.Tensor([1, 1, 1]))
        self.global_weight = config.global_weight
        self.mid_weight = config.mid_weight
        self.local_weight = config.local_weight
        logger.info(f'alpha = {self.global_weight}, beta = {self.mid_weight}, 1-alpha-beta = {self.local_weight}')

        self.mol_encoder = GCNModel(config.num_layer, config.emb_dim, self.dropout_ratio).to(self.device)

        self.sent_weight = nn.Linear(self.hidden_size, self.emb_dim)
        self.mtoken_weight = nn.Linear(self.hidden_size, self.emb_dim)
        self.token_weight = nn.Linear(self.hidden_size, self.emb_dim)

        self.mol_weight = nn.Linear(self.emb_dim, self.emb_dim)
        self.motif_weight = nn.Linear(self.emb_dim, self.emb_dim)
        self.atom_weight = nn.Linear(self.emb_dim, self.emb_dim)

        self.sent_ln = nn.LayerNorm(self.emb_dim).to(self.device)
        self.mol_ln = nn.LayerNorm(self.emb_dim).to(self.device)
        self.mid_ln = nn.LayerNorm(self.emb_dim, elementwise_affine=False).to(self.device)
        self.local_ln = nn.LayerNorm(self.emb_dim, elementwise_affine=False).to(self.device)

        self.params = list(self.parameters())

        self.text_encoder = BertModel.from_pretrained(text_encoder_path)
        if self.training:
            self.text_encoder.train()
        self.text_encoder = self.text_encoder.to(self.device)

    def forward(self, data_batch):
        sent_rep, mtoken_rep, token_rep, atom_rep, motif_rep, mol_rep, \
            token_mask, atom_mask, motif_mask = self.get_rep(data_batch)
        
        global_logits = self.get_global_logits(sent_rep, mol_rep)
        local_logits = self.get_local_logits(token_rep, atom_rep, token_mask, atom_mask)

        phrase_rep, phrase_mask = self.get_phrase_rep(mtoken_rep, token_mask, motif_rep, motif_mask)
        mid_logits = self.get_mid_logits(phrase_rep, motif_rep, phrase_mask, motif_mask)

        global_loss = cross_entropy_loss(global_logits)
        mid_loss = cross_entropy_loss(mid_logits)
        local_loss = cross_entropy_loss(local_logits)
        loss = self.combine_grain(global_loss, mid_loss, local_loss)

        return loss, sent_rep, mtoken_rep, token_rep, atom_rep, motif_rep, mol_rep, token_mask, atom_mask, motif_mask
        
    def get_rep(self, data_batch):
        input_ids, attention_mask = data_batch.input_ids, data_batch.attention_mask
        x, edge_index = data_batch.x, data_batch.edge_index
        num_part = data_batch.num_part
        
        # sentence-level, token-level feature
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True, output_attentions=False)
        
        sent_rep = text_out['pooler_output']
        sent_rep = self.sent_weight(sent_rep)  # [batch_size, emb_dim]
        sent_rep = self.sent_ln(sent_rep)
        sent_rep = sent_rep * torch.exp(self.temp)

        mtoken_rep = text_out['hidden_states'][-2]
        mtoken_rep = self.mtoken_weight(mtoken_rep)  # [batch_size, max_length, emb_dim]
        mtoken_rep = mtoken_rep * torch.exp(self.temp)

        token_rep = text_out['hidden_states'][-3]
        token_rep = self.token_weight(token_rep)  # [batch_size, max_length, emb_dim]
        token_rep = token_rep * torch.exp(self.temp)
        token_mask = attention_mask.bool()
        
        # molecule-level, motif-level, atom-level feature
        mol_out = self.mol_encoder(x, edge_index)
        mol_out = self.mol_ln(mol_out)
        atom_rep, motif_rep, mol_rep, atom_mask, motif_mask = group_node_rep(mol_out, 
                                    len(data_batch), num_part, self.num_atom, self.num_motif)
        
        atom_mask = atom_mask.to(self.device)
        motif_mask = motif_mask.to(self.device)

        mol_rep = self.mol_weight(mol_rep.to(self.device))
        mol_rep = mol_rep * torch.exp(self.temp)
        motif_rep = self.motif_weight(motif_rep.to(self.device))
        motif_rep = motif_rep * torch.exp(self.temp)
        atom_rep = self.atom_weight(atom_rep.to(self.device))
        atom_rep = atom_rep * torch.exp(self.temp)

        return sent_rep, mtoken_rep, token_rep, atom_rep, motif_rep, \
            mol_rep, token_mask, atom_mask, motif_mask
    
    def get_sparc_rep(self, token_rep, atom_rep, token_mask, atom_mask):
        '''
            token_rep: [batch_size, num_token, emb_dim]
            atom_rep: [batch_size, num_atom, emb_dim]
            token_mask: [batch_size, num_token]
            atom_mask: [batch_size, num_atom]
        '''
        if token_mask is not None and atom_mask is not None:
            # [batch_size, num_token, num_atom]
            sim_mask = torch.matmul(token_mask.float().unsqueeze(-1), 
                                    atom_mask.float().unsqueeze(-1).transpose(-2, -1))
        else:
            sim_mask = None
        sim_matrix = torch.einsum('btd,bad->bta', token_rep, atom_rep)  # [batch_size, num_token, num_atom]

        # min-max normalization
        min_sim = torch.min(sim_matrix, dim=-1, keepdim=True)[0]
        max_sim = torch.max(sim_matrix, dim=-1, keepdim=True)[0]
        sim_matrix = (sim_matrix - min_sim) / (max_sim - min_sim + self.epsilon)
        sim_matrix = sim_matrix.masked_fill(sim_mask == 0, self.epsilon)

        atom_align_weights = sim_matrix / (torch.sum(sim_matrix, dim=-1, keepdim=True) + self.epsilon)
        grouped_atom_rep = torch.einsum('bta,bad->btd', atom_align_weights, atom_rep)   # [batch_size, num_token, emb_dim]

        grouped_atom_rep_norm = F.normalize(grouped_atom_rep, p=2, dim=-1)      # [batch_size, num_token, emb_dim]
        token_rep_norm = F.normalize(token_rep, p=2, dim=-1)    # [batch_size, num_token, emb_dim]
        
        return token_rep_norm, grouped_atom_rep_norm
    
    def get_phrase_rep(self, mtoken_rep, token_mask, motif_rep, motif_mask):
        '''
            mtoken_rep: [batch_size, num_token, emb_dim]
            token_mask: [batch_size, num_token]
            motif_rep: [batch_size, num_motif, emb_dim]
            motif_mask: [batch_size, num_motif]
        '''
        x = mtoken_rep
        x_mask = token_mask
        y = motif_rep
        y_mask = motif_mask

        invert_x_mask = ~x_mask
        invert_y_mask = ~y_mask

        x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
        y = torch.nn.functional.normalize(y, p=2, dim=-1, eps=1e-12)
        tmp1 = torch.matmul(x, y.transpose(1, 2))
        tmp1 = tmp1.masked_fill(invert_x_mask.unsqueeze(-1), self.epsilon).masked_fill(invert_y_mask.unsqueeze(-2), self.epsilon)
        C = 1 - tmp1

        n = x.shape[1]
        m = y.shape[1]
        batch_size = x.shape[0]

        beta = 0.5
        sigma = (torch.ones([batch_size, m, 1]) / m).to(self.device)
        dist = torch.ones([batch_size, n, m]).to(self.device)
        A = torch.exp(-C / beta)

        for t in range(50):
            Q = A * dist
            delta = 1 / (n * torch.matmul(Q, sigma) + self.epsilon)
            sigma = 1 / (m * torch.matmul(Q.transpose(1, 2), delta) + self.epsilon)
            tmp = torch.bmm(torch.diag_embed(delta.squeeze(-1)), Q)
            dist = torch.bmm(tmp, torch.diag_embed(sigma.squeeze(-1)))
            
        ot = dist.masked_fill(invert_x_mask.unsqueeze(-1), 1.).masked_fill(invert_y_mask.unsqueeze(-2), 1.).min(dim=-1)[1]

        # ot: [batch_size, num_token]
        one_hot_ot = F.one_hot(ot, num_classes=self.num_motif).unsqueeze(-1)  # [batch_size, num_token, num_motif, 1]
        phrase_rep = mtoken_rep.unsqueeze(2) * one_hot_ot  # [batch_size, num_token, num_motif, emb_dim]
        phrase_rep = phrase_rep.sum(dim=1)  # [batch_size, num_motif, emb_dim]
        select_motif_nums = one_hot_ot.sum(dim=1) # [batch_size, num_token]
        phrase_rep = phrase_rep / (select_motif_nums + self.epsilon)
        phrase_mask = motif_mask

        return phrase_rep, phrase_mask

    def get_global_logits(self, sent_rep, mol_rep):
        '''
            sent_rep : [batch_size, emb_dim]
            mol_rep : [batch_size, emb_dim]
        '''
        logits = torch.matmul(sent_rep, mol_rep.t())   # [batch_size, batch_size]
        return logits
    
    def get_mid_logits(self, phrase_rep, motif_rep, phrase_mask=None, motif_mask=None):
        '''
            phrase_rep: [batch_size, num_motif, emb_dim]
            motif_rep: [batch_size, num_motif, emb_dim]
            phrase_mask: [batch_size, num_motif]
            motif_mask: [batch_size, num_motif]
        '''
        text_rep, grouped_motif_rep = self.get_sparc_rep(phrase_rep, motif_rep, phrase_mask, motif_mask)
        
        text_rep = torch.sum(text_rep, dim=1)
        text_rep = self.mid_ln(text_rep)
        
        grouped_motif_rep = torch.sum(grouped_motif_rep, dim=1)
        grouped_motif_rep = self.mid_ln(grouped_motif_rep)

        logits = torch.matmul(text_rep, grouped_motif_rep.permute(1, 0))

        return logits
    
    def get_local_logits(self, token_rep, atom_rep, token_mask=None, atom_mask=None):
        '''
            text_rep: [batch_size, num_token, emb_dim]
            atom_rep: [batch_size, num_atom, emb_dim]
            token_mask: [batch_size, num_token]
            atom_mask: [batch_size, num_atom]
        '''
        text_rep, grouped_atom_rep = self.get_sparc_rep(token_rep, atom_rep, token_mask, atom_mask)

        text_rep = torch.sum(text_rep, dim=1)
        text_rep = self.local_ln(text_rep)
        grouped_atom_rep = torch.sum(grouped_atom_rep, dim=1)
        grouped_atom_rep = self.local_ln(grouped_atom_rep)
        logits = torch.matmul(text_rep, grouped_atom_rep.permute(1, 0))

        return logits

    def combine_grain(self, global_grain, mid_grain, local_grain):
        if self.combine == 'weighted':
            return self.global_weight * global_grain + \
                self.mid_weight * mid_grain + self.local_weight * local_grain
        
        elif self.combine == 'molatom':
            return self.global_weight * global_grain + self.local_weight * local_grain
        
        elif self.combine == 'molmotif':
            return self.global_weight * global_grain + self.mid_weight * mid_grain
        
        elif self.combine == 'motifatom':
            return self.mid_weight * mid_grain + self.local_weight * local_grain

        elif self.combine == 'mol':
            return global_grain
        
        elif self.combine == 'motif':
            return mid_grain

        elif self.combine == 'atom':
            return local_grain