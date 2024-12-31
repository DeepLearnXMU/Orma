import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from transformers import BertTokenizerFast
from chemutils import get_mol, motif_decomp


class MolGraph(object):
  def __init__(self, cid, smiles, emb_path, graph_path):
    self.cid = cid
    self.smiles = smiles
    self.mol = get_mol(smiles)

    self.token_embs = np.load(emb_path, allow_pickle=True)[()]

    atom_x, atom_edge_index = [], []
    with open(f'{graph_path}/{self.cid}.graph', 'r') as f:
      # `f` means file `xxx.graph`, including the connection between edges and their substruct_id
      next(f)
      for line in f: # edges
        if line != "\n":
          edge = *map(int, line.split()), 
          atom_edge_index.append(edge)
        else:
          break
      next(f)
      for line in f: #get mol2vec features:
        # `self.gt.token_embs` means file `token_embedding_dict.npy`
        substruct_id = line.strip().split()[-1]
        if substruct_id in self.token_embs:
          atom_x.append(self.token_embs[substruct_id])
        else:
          atom_x.append(self.token_embs['UNK'])
      self.x_nosuper = torch.from_numpy(np.array(atom_x))
      self.edge_index_nosuper = torch.from_numpy(np.array(atom_edge_index)).permute(1, 0)

    # add super node, [119, 0] represents the super graph node
    num_atoms = self.mol.GetNumAtoms()

    # add motif, [120, 0] represents motif node
    self.cliques = motif_decomp(self.mol)
    num_motif = len(self.cliques)
    if num_motif > 0:
      # motif_x = torch.tensor([[120, 0]]).repeat_interleave(num_motif, dim=0).to(self.x_nosuper.device)

      motif_x, motif_edge_index = [], []
      for k, motif in enumerate(self.cliques):
        motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
        motif_x.append(torch.mean(self.x_nosuper[torch.from_numpy(np.array(motif))], dim=0))

      motif_x = torch.stack(motif_x).to(self.edge_index_nosuper.device)
      motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)

      super_x = torch.mean(motif_x, dim=0).unsqueeze(0)
      super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
      super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
      self.edge_index = torch.cat((self.edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

      self.x = torch.cat((self.x_nosuper, motif_x, super_x), dim=0)
      self.num_part = (num_atoms, num_motif, 1)

    else:
      super_x = torch.mean(self.x_nosuper, dim=0).unsqueeze(0)
      self.x = torch.cat((self.x_nosuper, super_x), dim=0)

      super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
      super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
      self.edge_index = torch.cat((self.edge_index_nosuper, super_edge_index), dim=1)

      self.num_part = (num_atoms, 0, 1)

  def size_node(self):
    return self.x.size()[0]

  def size_atom(self):
    return self.x_nosuper.size()[0]


class MolDataset(Dataset):
  def __init__(self, data_path, tokenizer_path, max_text_length, emb_path, graph_path):
    self.data = pd.read_csv(data_path, sep=',')

    self.max_text_length = max_text_length
    self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    self.emb_path = emb_path
    self.graph_path = graph_path

  def _tokenize(self, desc):
    output = self.tokenizer(desc, truncation=True, max_length=self.max_text_length,
                            padding='max_length', return_tensors='pt')
    
    input_ids = output['input_ids']
    token_type_ids = output['token_type_ids']
    attention_mask = output['attention_mask']

    return input_ids, token_type_ids, attention_mask
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    cid = self.data['CID'].values[idx]
    smiles = self.data['CanonicalSMILES'].values[idx]
    title = self.data['Title'].values[idx]
    desc = self.data['Description'].values[idx]

    input_ids, token_type_ids, attention_mask = self._tokenize(desc)
    mol_graph = MolGraph(cid, smiles, self.emb_path, self.graph_path)

    return input_ids, token_type_ids, attention_mask, mol_graph


def get_data(batch):
  '''
    Batch:
      input_ids, token_type_ids, attention_mask: tokenizer outputs
      graph_batch:
        x: [num_node, 300]
        edge_index: [2, num_edge], represents Graph connectivity
  '''
  data_batch = []
  for input_ids, token_type_ids, attention_mask, mol_graph in batch:
    data = Data(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                x=mol_graph.x, edge_index=mol_graph.edge_index, num_part=mol_graph.num_part, cliques=mol_graph.cliques)
    data_batch.append(data)

  new_batch = Batch().from_data_list(data_batch)
  return new_batch
