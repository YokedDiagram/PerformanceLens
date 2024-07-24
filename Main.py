# -*- coding: utf-8 -*-
import torch
from ModelTrainer import ModelTrainer, multi_quantile_loss
from PerformanceLens import PerformanceLensGNN, PerformanceLensHybrid
import warnings

warnings.filterwarnings('ignore')

#Set Environment Variables
data_dir = './MicroSS/'
model_choice = 'PerformanceLens-Hybrid'
predict_graph = False
validate_on_trace = False

if 'TrainTicket' in  data_dir:
    trainer_text = """
    ######################################################################
    ########### PerformanceLens Trained on Train Ticket Dataset ##########
    ######################################################################
    """
else:
    trainer_text = """
    ######################################################################
    ########### PerformanceLens Trained on MicroSS Dataset   #############
    ######################################################################
    """
print(trainer_text)

# Initialize Model Trainer
batch_size = 5
quantiles = [0.0013, 0.0062, 0.0228, 0.0668, 0.1587, 0.2266, 0.3085, 0.3539, 0.4013, 0.4503, 0.5000, 0.5498, 0.5987, 0.6462, 0.6915, 0.7734, 0.8413, 0.9332, 0.9772, 0.9938, 0.9987]
model_trainer = ModelTrainer(data_dir, batch_size, quantiles, predict_graph,\
                             validate_on_trace=validate_on_trace)

# Initialize the model
input_dim = model_trainer.graphs[0].x.size()[1] - 1
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 30
output_dim = len(quantiles)

if model_choice == 'PerformanceLens-GNN':
    model = PerformanceLensGNN(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
                predict_graph=model_trainer.predict_graph)
else:
    model = PerformanceLensHybrid(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
                predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
epochs = 10
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, multi_quantile_loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)
