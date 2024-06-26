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
5. Run PreprocessMetric.py file
6. Run PreprocessTraces.py file


