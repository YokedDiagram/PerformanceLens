# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:32:12 2024

@author: Hamza
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool
from torch.nn import Embedding

class PerformanceLensGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, embedding_dim, output_dim, predict_graph=True, pool='add'):
        super(PerformanceLensGNN, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.conv1 = GATConv(input_dim+embedding_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.predict_graph = predict_graph
        if self.predict_graph:
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = torch.nn.Linear(2 * hidden_dim, output_dim)
        self.pool = pool

    def forward(self, data, batch):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_index = x[:,-1].long()
        emb_vecs = self.embedding(node_index)
        assert x.shape[0] == emb_vecs.shape[0]
        x = torch.cat((x[:,:-1], emb_vecs), axis=1)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.gelu(x)
        if self.predict_graph == False:
            #Construct Edge Embeddings
            row, col = edge_index
            x = torch.cat([x[row], x[col]], dim = -1)
        else:
            if self.pool == 'mean':
                x = global_mean_pool(x, batch)
            else:
                x = global_add_pool(x, batch)
        # Fully connected layer
        x = self.fc(x)
        x = F.leaky_relu(x, 0.01)
        return x

class PerformanceLensHybrid(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, embedding_dim, output_dim, predict_graph=True):
        super(PerformanceLensHybrid, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.conv1 = GATConv(input_dim+embedding_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.gru_cell = GRUModel(hidden_dim, output_dim, output_dim)
        self.predict_graph = predict_graph
        self.output_dim = output_dim
        
    def forward(self, data, batch):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_index = x[:,-1].long()
        emb_vecs = self.embedding(node_index)
        assert x.shape[0] == emb_vecs.shape[0]
        x = torch.cat((x[:,:-1], emb_vecs), axis=1)
        #x = torch.cat((x[:,:-1], emb_vecs), axis=1)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.gelu(x)
        
        batch_edge = batch[edge_index[0]]
        row, col = edge_index
        #Construct Edge Embeddings
        x = torch.cat([x[row], x[col]], dim = -1)
        
        # Fully connected layer
        x = self.fc(x)
        x = F.gelu(x)
        
        #Construct GRU Input
        gru_input, mask, max_edges = construct_tensor(x, batch_edge)
      
        predictions, hidden_state = self.gru_cell(gru_input)
        
        # Apply mask to predictions
        masked_predictions = predictions.view(predictions.size(0), -1, self.output_dim) * mask.unsqueeze(2)
        
        if self.predict_graph:
            # Find the index of the last non-zero value in the max_nodes dimension for each sample
            last_non_zero_indices = torch.tensor([torch.nonzero(row).tolist()[-1][-1] if torch.sum(row) > 0 else 0 for row in mask])
    
            # Extract the corresponding prediction for each sample
            selected_predictions = masked_predictions[torch.arange(masked_predictions.size(0)), last_non_zero_indices]
        else:
            selected_predictions = masked_predictions[mask.bool()]
        return selected_predictions

# Define the complete GRU model
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = VanillaGRU(input_size, hidden_size)
        #self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        hidden = self.gru.init_hidden(batch_size)
        outputs = torch.zeros(batch_size, seq_length, self.hidden_size)
        for t in range(seq_length):
            hidden = self.gru(x[:, t, :], hidden)
            outputs[:, t, :] = hidden
        #output = self.fc(hidden)
        return outputs, hidden
    
# Define the GRU cell
class VanillaGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaGRU, self).__init__()
        self.hidden_size = hidden_size

        # Update gate
        self.Wz = torch.nn.Linear(input_size + hidden_size, hidden_size)
        
        # Reset gate
        self.Wr = torch.nn.Linear(input_size + hidden_size, hidden_size)
        
        # Candidate activation
        self.Wh = torch.nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        
        z = torch.sigmoid(self.Wz(combined))
        r = torch.sigmoid(self.Wr(combined))
        
        combined_reset = torch.cat((x, r * hidden), 1)
        h_tilde = F.leaky_relu(self.Wh(combined_reset), 0.01)
        
        hidden = (1 - z) * hidden + z * h_tilde
        return hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

def construct_tensor(x, batch):
    # Find unique values (graphs) and their counts
    unique_values, counts = torch.unique(batch, return_counts=True)
    batch_size = counts.size(0)
    max_nodes = counts.max().item()

    # Initialize the output tensor with zeros
    output = torch.zeros(batch_size, max_nodes, x.size(1))
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.int)

    # Iterate over unique values (graphs)
    for i, graph_id in enumerate(unique_values):
        # Get the indices of nodes belonging to the current graph
        graph_indices = (batch == graph_id).nonzero(as_tuple=True)[0]

        # Pad the graph's nodes with zeros to match the maximum count
        padded_nodes = x[graph_indices]
        padding = max_nodes - len(graph_indices)
        if padding != 0:
            padded_nodes = torch.cat([x[graph_indices], torch.zeros(max_nodes - len(graph_indices), x.size(1))])
        
        # Compute the mask tensor based on the condition
        mask[i, :len(graph_indices)] = 1
        # Assign the padded nodes to the output tensor
        output[i] = padded_nodes

    return output, mask, max_nodes
