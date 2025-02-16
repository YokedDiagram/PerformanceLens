# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import math

def prepare_data(path, normalize_features= [], normalize_by_node_features = [], scale_features = []):
    data = pd.DataFrame()
    data_dir = Path(path)
    file_list = list(map(str, data_dir.glob("*1011.pkl")))

    print("\n********************************")
    print("*********Loading Files**********")
    print("********************************\n")
    trace_to_integer = {}
    for data_file in tqdm(file_list):
        with open(data_file, 'rb') as file:
            file_data = pickle.load(file)
        df = pd.DataFrame(file_data)
        df['latency'] = df['latency'].apply(lambda latencies: micro_to_mili(latencies))
        df['max_latency'] = df['latency'].apply(lambda latencies: max(latencies))
        df = df[df['max_latency'] <= 20000]
        df = df[df['max_latency'] >= 1]
        df['original_latency'] = df['latency']
        df['timestamp'] = df['timestamp'].apply(lambda stamps: stamps_to_time(stamps))
        df['trace_integer'] = df.apply(lambda row: get_trace_integer(row, trace_to_integer), axis=1)
        data = pd.concat([data,df])
    
    counts = data['label'].value_counts()
    ##################################################
    print("\n***********Fault Distribution************")
    print(counts)
    print("*****************************************\n")
    ##################################################
    
    outliers = ['d3fdfb558dfb754de55b9e8d80eeb7a3', \
                'f8398b6b1ad61f915ff275141eb345e7', \
                'd50503eb258fcf371b719b716555f55d', \
                '9f6fb14ccb19fc48668c1898c4835905', \
                '6b479c5de1a70eb50b1ea151c93b6181']
    data = data[~data['trace_id'].isin(outliers)]
    measures = {}
    data = data.reset_index(drop=True)
    transformation_features = normalize_by_node_features + normalize_features + scale_features
    for feature in transformation_features:
        measures[feature] = {}
    for feature in scale_features:
        data, feature_max, feature_min = scale(data, feature)
        measures[feature]['scale'] = [feature_max, feature_min]
    for feature in normalize_by_node_features:
        data, feature_measures = normalize_by_node(data, feature)
        measures[feature]['norm_by_node'] = feature_measures
    for feature in normalize_features:
        data, feature_mean, feature_std = normalize(data, feature)
        measures[feature]['norm'] = [feature_mean, feature_std]
    data, stats = normalize_by_trace(data)
    
    for column in ['mean', 'std', 'maximum', 'minimum']:
        data, feature_mean, feature_std = normalize(data, column)
    
    global_map = prepare_global_map(data)
    return data, global_map, measures

def order_data(data_row):
    latencies = data_row['original_latency']
    sorted_indices = sorted(range(len(latencies)), key=lambda i: latencies[i])
    for column in data_row.index:
        if isinstance(data_row[column], list):
            data_row[column] = [data_row[column][i] for i in sorted_indices]
    data_row['trace_length'] = len(sorted_indices)
    return data_row

def normalize_by_trace(data, grouped_data=True):    
    if grouped_data:
        values = data.apply(pd.Series.explode)
    else:
        values = data
        
    print("\n********************************")
    print("***Normalizing latency by trace***")
    print("********************************\n")
   
    # Group by 'node_name' and calculate mean and std of column values
    result = values.groupby(['trace_integer', 's_t'])['latency'].agg(['mean', 'std', 'max', 'min', 'count'])
    result = result.reset_index()
    result[['source', 'target']] = result['s_t'].apply(pd.Series)
    result.drop(columns=['s_t'], inplace=True)
    result.set_index(['trace_integer', 'source', 'target'], inplace=True)
    result.fillna(1, inplace=True)
    
    columns = ['latency']
    values = values.groupby(['trace_integer', 's_t']).apply(normalize_cluster, columns=columns)
    values = values.reset_index(drop=True)
    if grouped_data:
        print("Grouping by trace_id")
        values = values.groupby(['trace_id', 'trace_integer']).agg(lambda x: x.tolist())
        values = values.reset_index()
        print("Ordering Rows")
        values = values.apply(lambda row: order_data(row), axis=1)
    return values, result

def normalize_cluster(cluster, columns):
    data = cluster
    for column in columns:
        mean = data[column].mean()
        std = data[column].std()
        if column == 'latency':
            count = data[column].count()
            maximum = data[column].max()
            minimum = data[column].min()
            cluster['mean'] = mean
            cluster['maximum'] = maximum
            cluster['minimum'] = minimum
            cluster['count'] = count
        if std == 0 or np.isnan(std):
            cluster['std'] = 1
        else:
            cluster['std'] = std
    return cluster

def recover_by_cluster(values, trace_integers, measures):
    recovered_values = []
    for (value, trace_integer) in zip(values, trace_integers):
        mean = measures.loc[trace_integer.item(), 'mean']
        std = measures.loc[trace_integer.item(), 'std']
        recovered_value = [(value * std) + mean]
        recovered_values = recovered_values + [recovered_value]
    return torch.tensor(recovered_values, dtype=torch.float32)

def get_trace_integer(row, trace_to_integer):
    s_t = row['s_t']
    trace = frozenset(s_t)
    if trace not in trace_to_integer:
        next_integer = max(trace_to_integer.values()) + 1 if trace_to_integer else 0
        trace_to_integer[trace] = next_integer
        next_integer += 1
    return trace_to_integer[trace]

def stamps_to_time(stamps):
    time = []
    for stamp in stamps:
        time = time + [datetime.fromtimestamp(stamp/1000000)\
                       .strftime("%d/%m/%Y %H:%M:%S.%f")]
    return time

def micro_to_mili(latencies):
    return [latency / 1000 for latency in latencies]

def scale(data, column):
    values = pd.DataFrame()
    values = data.explode(column)
    
    maximum = values[column].max()
    minimum = values[column].min()
    data[column] = data[column].apply(lambda row: scale_values(row))
    return data, maximum, minimum

def log(value):
    if value != 0:
        scaled_value = np.log10(value)
    else:
        scaled_value = 0
    return scaled_value

def scale_values(values):
    scaled_values = []
    for value in values:
        scaled_value = log(value)
        scaled_values = scaled_values + [scaled_value]
    return scaled_values

def normalize_by_edge(data, column, grouped_data=True):
    values = pd.DataFrame()
    filtered_data = data[[column, 's_t']].copy()
    
    if grouped_data:
        values[column] = data[column].explode(column)
        df_edges = data['s_t'].explode('s_t')
    else:
        values[column] = data[column]
        df_edges = data['s_t']
    values['edges'] = df_edges
    
    print("\n********************************")
    print("***Normalizing latency by node***")
    print("********************************\n")
   
    # Group by 'node_name' and calculate mean and std of column values
    result = values.groupby('edges')[column].agg(['mean', 'std'])

    # Rename the columns for clarity
    result.columns = ['average', 'std_dev']
    
    result = pd.DataFrame(result)
    result.index = pd.MultiIndex.from_tuples(result.index)
    result.fillna(1, inplace=True)
    result['std_dev'].replace(0,1, inplace=True)
    
    data[column+'_normalized'] = filtered_data.progress_apply(lambda row: centre_by_group(row[column], row['s_t'], result, grouped_data), axis=1)
    return data, result

def normalize_by_node(data, column, grouped_data=True):
    tqdm.pandas()
    values = pd.DataFrame()
    filtered_data = data[[column, 's_t']].copy()
    if grouped_data:
        values[column] = data[column].explode(column)
        df_edges = data['s_t'].explode('s_t')
    else:
        values[column] = data[column]
        df_edges = data['s_t']
    nodes = df_edges.apply(lambda x: x[1])
    values['node_name'] = nodes
    
    print("\n********************************")
    print("***Normalizing " + column + " by node***")
    print("********************************\n")
   
    # Group by 'node_name' and calculate mean and std of column values
    result = values.groupby('node_name')[column].agg(['mean', 'std'])

    # Rename the columns for clarity
    result.columns = ['average', 'std_dev']
    result = pd.DataFrame(result)
    result.fillna(1, inplace=True)
    result['std_dev'].replace(0,1, inplace=True)
    
    data[column+'_normalized'] = filtered_data.progress_apply(lambda row: centre_by_group(row[column], row['s_t'], result, grouped_data), axis=1)
    return data, result

def centre_by_group(values, edges, measures, grouped_data=True):
    if grouped_data:
        centred_values = []
        for (value, edge) in zip(values, edges):
            #Check if we need to center by edges or by node
            if measures.index.nlevels != 2:
                node = edge[1]
            mean = measures.loc[node, 'average']
            std = measures.loc[node, 'std_dev']
            centred_value = [(value - mean) / std]
            centred_values = centred_values + centred_value
    else:
        node = edges[1]
        mean = measures.loc[node, 'average']
        std = measures.loc[node, 'std_dev']
        centred_values = (values - mean) / std
    
    return centred_values

def normalize(data, column, grouped_data=True):
    if grouped_data:
        total = 0
        count = 0
        squared_sum = 0
        for values in data[column]:
            # Update total by summing up all values in each list
            total += sum(values)
            # Update count by adding the length of each list
            count += len(values)
            # Update squared_sum by summing up the squares of all values in each list
            squared_sum += sum(map(lambda x: x**2, values))
        
        # Calculate the mean
        mean = total / count
        
        # Calculate the mean of squares
        squared_mean = squared_sum / count
        
        # Calculate the standard deviation
        std = math.sqrt(squared_mean - mean**2)
    
        data[column] = data[column].apply(lambda values: centre(values, mean, std, column))
    else:
        mean = data[column].mean()
        std = data[column].std()
        data[column] = (data[column] - mean) / std
    return data, mean, std

def centre(values, mean, std, column):
    centred_values = []
    if std == 0: std=1
    for value in values:
        centred_value = (value - mean) / std
        centred_values = centred_values + [centred_value]
    return centred_values    

def prepare_global_map(data):
    global_map ={}
    for s_t in data['s_t']:
        edges = pd.DataFrame(s_t, columns = ['source', 'target'])
        unique_nodes = pd.concat([edges['source'], edges['target']]).unique()
        for node in unique_nodes:
            if global_map == {}:
                global_map[node] = 0
            elif node not in global_map.keys():
                global_map[node] = max(global_map.values()) + 1
    return global_map

def prepare_graph(trace, global_map, normalize_by_node_features = []):
    nodes = {'cpu_use': trace['cpu_use'], \
             'mem_use_percent': trace['mem_use_percent'],
             'mem_use_amount': trace['mem_use_amount'],
             'net_send_rate': trace['net_send_rate'],
             'net_receive_rate': trace['net_receive_rate'],
             'file_read_rate': trace['file_read_rate']}
        

    for feature in normalize_by_node_features:
        if feature != 'latency':
            nodes[feature+'_normalized'] = trace[feature+'_normalized']
    
    nodes = pd.DataFrame(nodes)

    #Create dataframe of edges
    edges = pd.DataFrame(trace['s_t'], columns = ['source', 'target'])

    edge_attr = {'mean': trace['mean'],\
                 'std': trace['std'],\
                 'max': trace['maximum'],\
                 'min': trace['minimum']}  
    
    edge_attr = pd.DataFrame(edge_attr)
    #Assume that the metrics belong to the target node in the edge. Store the 
    #node name of the target with the metrics
    nodes['node_name'] = edges['target']
    
    y_edge_features = {'latency': trace['latency']}
    y_edge_features = pd.DataFrame(y_edge_features)
    
    original = {'latency': trace['original_latency']}
    original = pd.DataFrame(original)
    

    trace_lat = y_edge_features['latency'].max()
    #Find all unique node names
    unique_nodes = pd.concat([edges['source'], edges['target']]).unique()
    
    #Check nodes that only occur as source in edges, therefore will have no metric
    #values in the node feature matrix.
    not_in_target = edges[~edges['source'].isin(edges['target'])]

    if len(not_in_target) > 0:
    # Create a DataFrame filled with zeros with the same number of rows as 'not_in_target'
    # and one less column than 'nodes'
        zero_df = pd.DataFrame(0, index=range(len(not_in_target)), columns=nodes.columns[:-1])
        #zero_df['trace_integer'] = trace_integer
        source_df = pd.DataFrame(not_in_target['source'].values, columns=['node_name'])
        
        # Reset the index of zero_df and source_df to align them properly
        zero_df.reset_index(drop=True, inplace=True)
        source_df.reset_index(drop=True, inplace=True)
        # Add 'node_name' column to zero_df and populate it with values from 'source' column of not_in_target
        zero_df = pd.concat([zero_df, source_df], axis = 1)
        # Append zero_df to the bottom of nodes DataFrame
        nodes = pd.concat([nodes, zero_df])

    nodes = nodes.groupby('node_name').mean().reset_index()
    node_name_column = nodes.pop('node_name')  # Remove 'node_name' from the DataFrame
    nodes['node_name'] = node_name_column  # Add 'node_name' as the last column
    nodes = nodes.reset_index(drop=True)
    
    #Map node names to integers
    node_to_int = {node: i for i, node in enumerate(unique_nodes)}
    nodes['node_id'] = nodes['node_name'].map(node_to_int)
    edges['source'] = edges['source'].map(node_to_int)
    edges['target'] = edges['target'].map(node_to_int)
    
    nodes = nodes.sort_values(by='node_id')
    nodes = nodes.drop(columns=['node_id'])
    
    nodes['node_name'] = nodes['node_name'].map(global_map)
    
    #Convert to tensors
    nodes_tensor = torch.tensor(nodes.values, dtype=torch.float32)
    edges_tensor = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
    y_edge_tensor = torch.tensor(y_edge_features.values, dtype=torch.float32).squeeze(dim=1)
    original = torch.tensor(original.values, dtype=torch.float32)
    trace_lat_tensor = torch.tensor(trace_lat, dtype=torch.float32)
    edge_attr_tensor = torch.tensor(edge_attr.values, dtype=torch.float32)
    graph = Data(x=nodes_tensor, edge_index=edges_tensor, edge_attr=edge_attr_tensor, y=y_edge_tensor, trace_lat=trace_lat_tensor)
    graph.original = original
    graph.label = torch.tensor(trace['label'][0], dtype=torch.long)

    return graph

def preprocess(path):
    tqdm.pandas()
    normalize_features = ['cpu_use', 'mem_use_percent', 'mem_use_amount', 'net_send_rate', 'net_receive_rate', 'file_read_rate']
    normalize_by_node_features = ['cpu_use', 'mem_use_percent', 'mem_use_amount', 'net_send_rate', 'net_receive_rate', 'file_read_rate']
    scale_features = ['latency']
    data, global_map, measures = prepare_data(path, normalize_features, normalize_by_node_features, scale_features)
    print("\n********************************")
    print("********Preparing Graphs**********")
    print("********************************\n")
    graphs = data.progress_apply(lambda trace: prepare_graph(trace, global_map, normalize_by_node_features), axis=1)
    graphs = graphs.to_list()
    return data, graphs, global_map, measures
