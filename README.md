# PefrormanceLens

### Fine-Grained Latency Distribution Prediction for Microservice Call Chains in Cloud-Native Applications
PerformanceLens is a Neural Network model that utilizes a hybrid Graph Neural Network/Gated Recurrent Unit architecture to predict latency distribution of an API request. Not only do we predict end-to-end latency distribution but also the distribution of individual microservice calls involved in an API request.

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
**1.** Install the following packages:

      a. matplotlib==3.8.0
      
      b. numpy==2.0.1

      c. pandas==2.2.2
      
      d. torch==2.3.0
      
      e. torch_geometric==2.5.3
      
      f. torchmetrics==1.4.0.0
      
      g. tqdm==4.65.0

      h. scikit_learn==1.2.2 (Required for AnomalyTrainer.py)
      
**2.** Open Main.py file

**3.** Set environment variables such as data_dir, model_choice, predict_graph, and validate_on_trace
   
      a. data_dir can be set to either './TrainTicket/' or './MicroSS/' depending on the dataset to be used.
   
      b. model_choice can be set to either 'PerformanceLens-GNN' or 'PerformanceLens-Hybrid' depending on whether PerformanceLens should be run with or without the GRU block.
   
      c. predict_graph can be either True or False. If set to True, the model will only predict end-to-end latencies.
   
      d. validate_on_trace can be either True or False. It can only be set to True if predict_graph is set to False. If set to True, the validation results will be 
      based on end-to-end latencies only.

      e. The data files to be used can be changed based on the file_list variable (line 21) in Preprocess.py file
**4.** Run the Main.py file.

### Extras
Anomaly Detection folder contains data constructed using output of PerformanceLENS on Train Ticket dataset. This dataset can be used to perform trace based anomaly detection using the latency distribution output of PerformanceLENS.
AnomalyTrainer.py file can be run to reproduce the Anomaly Detection experiment on the created dataset. In the experiment we use traditional ML models trained on PerformanceLENS output to detect trace based anomalies.

