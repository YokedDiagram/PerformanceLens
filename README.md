# Pred X

### Multi-modal Performance Prediction in Microservices Based Applications
PredX is a Neural Network model that utilizes a hybrid Graph Neural Network/Gated Recurrent Unit architecture to predict latency distribution of an API request. Not only do we predict end-to-end latency distribution but also the distribution of individual microservice calls involved in an API request.

### Datasets
This repository provides a way to reproduce the results on two benchmark datasets: Train Ticket and MicroSS.

**Train Ticket**: Train Ticket dataset is uploaded along with the code in this repository. 

**MicroSS**: Dataset needs to be downloaded from the original publisher (https://github.com/CloudWise-OpenSource/GAIA-DataSet/tree/main). Follow the steps below for preprocessing MicroSS dataset for PredX.

1. Download metric and trace folders from: https://github.com/CloudWise-OpenSource/GAIA-DataSet/tree/main/MicroSS
2. Extract the zip files for metrics and traces.
3. Copy the raw csv files for metrics to ./MicroSS/metric_split/metric folder
4. Copy the raw csv files for traces to ./MicroSS/trace_split/trace
5. Run PreprocessMetric.py file in ./MicroSS directory
6. Run PreprocessTraces.py file in ./MicroSS directory

### Usage
1. Install the following packages:
   
   a. torch
   
   b. torch_geometric
   
   c. torchmetrics
   
   d. tqdm
   
   e. pandas
   
   f. numpy
2. Open Main.py file
3. Set environment variables such as data_dir, model_choice, predict_graph, and validate_on_trace
   
   a. data_dir can be set to either './TrainTicket/' or './MicroSS/' depending on the dataset to be used.
   
   b. model_choice can be set to either 'PredX-GNN' or 'PredX-Hybrid' de[ending on whether PredX should be run with or without the GRU block.
   
   c. predict_graph can be either True or False. If set to True, the model will only predict end-to-end latencies.
   
   d. validate_on_trace can be either True or False. It can only be set to True if predict_graph is set to False. If set to True, the validation results will be 
      based on end-to-end latencies only.
4. Run the Main.py file.



