# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:21:41 2024

@author: Hamza
"""
import sys
sys.path.append('./MicroSS')
import torch
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchmetrics.functional.regression import explained_variance
from torchmetrics import MeanAbsolutePercentageError
#Custom Imports
from CustomDataset import CustomDataset
from Preprocess import preprocess
from GraphConstructor import process_micross
import matplotlib.pyplot as plt

class ModelTrainer():
    def __init__(self, path, batch_size, quantiles=[], predict_graph=True, \
                 validate_on_trace=False):
        
        self.batch_size=batch_size
        self.quantiles=quantiles
        self.predict_graph=predict_graph
        self.validate_on_trace=validate_on_trace
        
        assert not(self.predict_graph and self.validate_on_trace)
        
        self.path = path
        if 'TrainTicket' in  path:
            #Pass the directory that contains data as pickle files to the preprocessing function
            data, graphs, global_map, measures = preprocess(path)
        else:
            data, graphs, global_map, measures = process_micross(path)
        
        if 'latency' in measures: measures = measures['latency']
        dataset = CustomDataset(graphs)
    
        # Split the dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create DataLoaders for training and validation
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.data = data
        self.measures = measures
        self.global_map  = global_map
        self.graphs = graphs
        
    def set_model(self, model):
        self.model = model
    
    def train(self, epochs, loss_fn, criterion, optimizer):
        train_loader = self.train_loader
        val_loader = self.val_loader
        # Training loop
        for epoch in range(1, epochs+1):
            self.model.train(True)
            total_loss = 0
            total_crit = 0
            for batch in train_loader:
                optimizer.zero_grad()
                recovered, recov_pred, loss, crit = self.step(batch, loss_fn, criterion)
                loss.backward()
                optimizer.step()
                total_crit += crit.item() 
                total_loss += loss.item()
                train_crit = total_crit/len(train_loader)
                train_loss = total_loss/len(train_loader)
                
            self.model.eval()
            total_val_loss = 0
            total_val_crit = 0
            with torch.no_grad():
                target = torch.tensor([])
                predictions = torch.tensor([])
                for batch in val_loader:
                    recovered, recov_pred, loss, crit = self.step(batch, loss_fn, criterion)
                    total_val_loss += loss.item()
                    total_val_crit += crit.item()
                    val_crit = total_val_crit/len(val_loader)
                    val_loss = total_val_loss/len(val_loader)
                    target = torch.cat([target, recovered], axis=0)
                    predictions = torch.cat([predictions, recov_pred], axis=0)
            cov_prob = coverage_probability(target, predictions)
            print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train criterion: {train_crit:.4f}, Val Loss: {val_loss:.4f}, Val criterion: {val_crit:.4f}, Val Cov Prob: {cov_prob:.4f}")
            calculate_metrics(self.quantiles, target, predictions, epoch, epochs)
            print("\n")
        return self.model
    
    def validate(self, loss_fn, criterion):
        self.model.eval()
        total_val_loss = 0
        total_val_crit = 0
        with torch.no_grad():
            target = torch.tensor([])
            predictions = torch.tensor([])
            for batch in self.val_loader:
                recovered, recov_pred, loss, crit = self.step(batch, loss_fn, criterion)
                total_val_loss += loss.item()
                total_val_crit += crit.item()
                val_crit = total_val_crit/len(self.val_loader)
                val_loss = total_val_loss/len(self.val_loader)
                target = torch.cat([target, recovered], axis=0)
                predictions = torch.cat([predictions, recov_pred], axis=0)
        return target, predictions
    
    def step(self, batch, loss_fn, criterion):
        recov_pred = self.model(batch, batch.batch)
        if self.predict_graph:
            recovered = batch.trace_lat
        else:
            recovered = batch.y
        target = torch.stack([recovered for _ in range(len(self.quantiles))], dim=1)
        loss = loss_fn(recov_pred, target, self.quantiles)
        if self.validate_on_trace:
            edge_index = batch.edge_index
            batch_nodes = batch.batch
            batch_edge = batch_nodes[edge_index[0]]
            recovered, recov_pred = self.extract_trace_lat(recovered, recov_pred, batch_edge)
        index = self.quantiles.index(0.5)
        crit = criterion(recov_pred[:,index], recovered)
        return recovered, recov_pred, loss, crit
    
    def extract_trace_lat(self, recovered, recov_pred, batch):
        last_indices = torch.bincount(batch)
        last_indices = torch.cumsum(last_indices, dim=0) - 1
        recovered = recovered[last_indices]
        recov_pred = recov_pred[last_indices]
        return recovered, recov_pred
    
    def predict(self, graph_idx):
        graph = self.graphs[graph_idx]
        if self.predict_graph:
            recovered = [graph.trace_lat]
        else:
            recovered = graph.y
            
        with torch.no_grad():
            recov_pred = self.model(graph, torch.zeros(graph.x.size(0), dtype=torch.int64))
                    
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
        
        return recov_pred

def calculate_metrics(quantiles, target, predictions, epoch, epochs):
    for i, quantile in enumerate(quantiles):
        mape = MAPE(predictions[:,i], target)
        e_var = explained_variance(predictions[:,i], target)
        qloss = quantile_loss(predictions[:,i], target, quantile)
        print('***************************************************************')
        print(f"Quantile: {quantile}, Quantile Loss: {qloss}")
        print(f"Val MAPE: {mape:.4f}, Exp Var: {e_var:.4f}")
        p_qloss = regional_quantile_loss(predictions[:,i], target, quantile)
        print(f"Quantile Loss by regions: {', '.join(f'{tensor.item():.4f}' for tensor in p_qloss.values())}")
        p_mape = regional_mape(target, predictions[:,i])
        print(f"MAPE by regions: {', '.join(f'{tensor.item():.4f}' for tensor in p_mape.values())}")
        p_mae = regional_mae(target, predictions[:,i])
        print(f"MAE by regions: {', '.join(f'{tensor.item():.4f}' for tensor in p_mae.values())}")
    if epoch == epochs:
        r = get_latency_regions(target, predictions)
        p_values = r[5]['p']
        print("\n")
        print("Percentile Values In Validation Data")
        print(p_values)

def regional_mape(target, predictions):
    r = get_latency_regions(target,predictions)
    
    m_1 = MAPE(torch.tensor(r[1]['y']),torch.tensor(r[1]['x']))
    m_2 = MAPE(torch.tensor(r[2]['y']),torch.tensor(r[2]['x']))
    m_3 = MAPE(torch.tensor(r[3]['y']),torch.tensor(r[3]['x']))
    m_4 = MAPE(torch.tensor(r[4]['y']),torch.tensor(r[4]['x']))
    m_5 = MAPE(torch.tensor(r[5]['y']),torch.tensor(r[5]['x']))
    
    return {1: m_1, 2: m_2, 3: m_3, 4: m_4, 5:m_5}

def regional_mae(target, predictions):
    r = get_latency_regions(target,predictions, rescale=True)
    
    m_1 = MAE(torch.tensor(r[1]['y']),torch.tensor(r[1]['x']))
    m_2 = MAE(torch.tensor(r[2]['y']),torch.tensor(r[2]['x']))
    m_3 = MAE(torch.tensor(r[3]['y']),torch.tensor(r[3]['x']))
    m_4 = MAE(torch.tensor(r[4]['y']),torch.tensor(r[4]['x']))
    m_5 = MAE(torch.tensor(r[5]['y']),torch.tensor(r[5]['x']))
    
    return {1: m_1, 2: m_2, 3: m_3, 4: m_4, 5:m_5}

def regional_quantile_loss(predictions, target, t_value):
    r = get_latency_regions(target, predictions)
    q_1 = quantile_loss(torch.tensor(r[1]['y']),torch.tensor(r[1]['x']))
    q_2 = quantile_loss(torch.tensor(r[2]['y']),torch.tensor(r[2]['x']))
    q_3 = quantile_loss(torch.tensor(r[3]['y']),torch.tensor(r[3]['x']))
    q_4 = quantile_loss(torch.tensor(r[4]['y']),torch.tensor(r[4]['x']))
    q_5 = quantile_loss(torch.tensor(r[5]['y']),torch.tensor(r[5]['x']))
    
    return {1: q_1, 2: q_2, 3: q_3, 4: q_4, 5:q_5}

def coverage_probability(targets, predictions):
    # Check if the target falls within the predicted range for latency distribution
    covered = [predictions[j, 0] <= targets[j] <= predictions[j, -1] for j in range(len(targets))]
    
    # Calculate coverage probability
    coverage_prob = sum(covered) / len(covered)
    
    # Return coverage probabilities
    return coverage_prob

def get_latency_regions(x,y, rescale=False):
    # Store the original values
    o_x = x
    o_y = y
    
    # Recover original scale of values
    x = 10 ** x
    y = 10 ** y
    
    x = x.numpy()
    y = y.numpy()

    percentile_10 = np.percentile(x, 10)
    percentile_25 = np.percentile(x, 25)
    percentile_50 = np.percentile(x, 50)
    percentile_75 = np.percentile(x, 75)
    percentile_90 = np.percentile(x, 90)
    percentile_95 = np.percentile(x, 95)
    
    p_values = [percentile_10, percentile_25, percentile_50, percentile_75, percentile_90, percentile_95]
    
    index_1 = np.where((x < 1))[0]
    index_2 = np.where((x > 1) & (x <= 50))[0]
    index_3 = np.where((x > 50) & (x <= 100))[0]
    index_4 = np.where((x > 100) & (x <= 1000))[0] 
    index_5 = np.where((x > 1)[0])
    
    if rescale:
        x_1 = x[index_1]
        y_1 = y[index_1]
        
        x_2 = x[index_2].flatten()
        y_2 = y[index_2].flatten()
        
        x_3 = x[index_3].flatten()
        y_3 = y[index_3].flatten()
        
        x_4 = x[index_4].flatten()
        y_4 = y[index_4].flatten()
        
        x_5 = x[index_5].flatten()
        y_5 = y[index_5].flatten()
    else:
        x_1 = o_x[index_1]
        y_1 = o_y[index_1]
        
        x_2 = o_x[index_2].flatten()
        y_2 = o_y[index_2].flatten()
        
        x_3 = o_x[index_3].flatten()
        y_3 = o_y[index_3].flatten()
        
        x_4 = o_x[index_4].flatten()
        y_4 = o_y[index_4].flatten()
        
        x_5 = o_x[index_5].flatten()
        y_5 = o_y[index_5].flatten()
        
    regions = {}
    # Slice values based on percentiles
    r_1 = {'x': x_1, 'y': y_1, 'p': p_values}
    regions[1] = r_1
    
    r_2 = {'x': x_2, 'y': y_2, 'p': p_values}
    regions[2] = r_2
    
    r_3 = {'x': x_3, 'y': y_3, 'p': p_values}
    regions[3] = r_3
    
    r_4 = {'x': x_4, 'y': y_4, 'p': p_values}
    regions[4] = r_4
    
    r_5 = {'x': x_5, 'y': y_5, 'p': p_values}
    regions[5] = r_5

    return regions
    
def MAPE(output, target):
    mean_abs_percentage_error = MeanAbsolutePercentageError()
    mape = mean_abs_percentage_error(output, target)
    return mape

def multi_quantile_loss(preds, target, quantiles):
    assert isinstance(preds, torch.Tensor), "Predictions must be a torch.Tensor"
    assert isinstance(target, torch.Tensor), "Target must be a torch.Tensor"
    assert isinstance(quantiles, (list, torch.Tensor)), "Quantiles must be a list or torch.Tensor"
    assert len(preds.shape) == 2, "Predictions must have 2 dimensions (batch_size, num_quantiles)"
    assert preds.shape[1] == len(quantiles), f"Number of predictions ({preds.shape[1]}) must match the number of quantiles ({len(quantiles)})"
    assert preds.shape == target.shape, "Shape of predictions must match shape of target"

    if isinstance(quantiles, list):
        assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
    else:
        assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"

    # Convert quantiles to a tensor if it's a list
    if isinstance(quantiles, list):
        quantiles_tensor = torch.tensor(quantiles, device=preds.device).view(1, -1)
    else:
        quantiles_tensor = quantiles.view(1, -1)

    # Calculate errors
    errors = (target - preds)
    
    # Calculate losses for each quantile
    losses = torch.max((quantiles_tensor - 1) * errors, quantiles_tensor * errors)

    # Sum the losses and take the mean
    loss = torch.mean(torch.sum(losses, dim=1))

    return loss

def MAE(output, target):
    criterion = torch.nn.L1Loss(reduction='mean')
    MAE = criterion(output, target)
    return MAE

def quantile_loss(preds, target, quantile=0.5):
    assert 0 < quantile < 1, "Quantile should be in (0, 1) range"
    errors = target - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.abs(loss).mean()
