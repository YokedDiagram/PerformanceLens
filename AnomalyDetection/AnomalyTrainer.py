# -*- coding: utf-8 -*-

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
}

train_data = pd.read_csv('train_anomalies.csv')
train_labels = train_data['label']

counts = train_data['label'].value_counts()
##################################################
print("\n***********Fault Distribution************")
print(counts)
print("*****************************************\n")

train_data = train_data.iloc[:,:-1]
train_data = torch.tensor(train_data.values, dtype=torch.float32)
interval_length = train_data[:, -1] - train_data[:, 0]
train_data = torch.cat((train_data, interval_length.unsqueeze(1)), dim=1)
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

val_data = pd.read_csv('validation_anomalies.csv')
val_labels = val_data['label']

counts = val_data['label'].value_counts()
##################################################
print("\n***********Fault Distribution************")
print(counts)
print("*****************************************\n")
val_data = val_data.iloc[:,:-1]
val_data = torch.tensor(val_data.values, dtype=torch.float32)
interval_length = val_data[:, -1] - val_data[:, 0]
val_data = torch.cat((val_data, interval_length.unsqueeze(1)), dim=1)
val_data = scaler.transform(val_data)

for name, clf in classifiers.items():
    data = train_data
    labels = torch.tensor(train_labels.values, dtype=torch.long)
    clf.fit(data, labels)
    train_predictions = clf.predict(data)
    accuracy = accuracy_score(labels, train_predictions)
    precision = precision_score(labels, train_predictions, average='macro')
    recall = recall_score(labels, train_predictions, average='macro')
    f1 = f1_score(labels, train_predictions, average='macro')
    count_of_ones = sum(pred == 1 for pred in train_predictions)  # Count occurrences of 1
    print(f'Training Metrics for {name}')
    print(f'Count of Ones: {count_of_ones}')
    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}')
    
    data = val_data
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(val_labels.values, dtype=torch.long)
    val_predictions = clf.predict(data)
    accuracy = accuracy_score(labels, val_predictions)
    precision = precision_score(labels, val_predictions, average='macro')
    recall = recall_score(labels, val_predictions, average='macro')
    f1 = f1_score(labels, val_predictions, average='macro')
    count_of_ones = sum(pred == 1 for pred in val_predictions)  # Count occurrences of 1
    print(f'Validation Metrics for {name}')
    print(f'Count of Ones: {count_of_ones}')
    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}')

    print('\n')
    print('******************************************')
    print('\n')
