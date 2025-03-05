# Graph-Neural-Network-for-Financial-Asset-Return-Prediction
This repository contains the code for a project that predicts future returns of financial assets by modeling their dynamic interactions using graph neural networks. The model leverages historical dynamic graphs, where each graph represents the financial network at a specific time step, to forecast asset excess returns in the next period.

Problem Context and Motivation
Financial markets are inherently complex, with asset prices and returns influenced by multiple interdependent factors. These interdependencies arise from various dimensions such as supply chains, industry sectors, return similarities, volatility spillovers, and overall market conditions.

The financial network is naturally modeled as a graph:

Nodes:
Each node represents a financial asset (a total of 372 assets). Each asset is characterized by 93 features that capture fundamental company information (e.g., financial ratios, cash flow metrics, and performance indicators). The set of nodes remains consistent over time, meaning every asset is present at each time step.

Edges:
Edges depict the relationships between assets. An edge is drawn between two assets if there is a long-term interrelationship between them, with the edge weight defined as the Pearson correlation of their historical excess returns over the past 36 months (ranging from -1 to 1). These connections are dynamic, reflecting shifts in market conditions and corporate strategies.

For example:

Lehman Brothers was removed from the network in October 2008 following its bankruptcy.
Tesla was added to the network in June 2010 with its IPO.
Apple's strategic pivot in 2019, marked by the launch of Apple TV+, transformed its market role and interconnections.
![image](https://github.com/user-attachments/assets/b4fd0230-be84-4643-9362-54d132a3bc65)


Project Goal
The primary goal is to develop a GNN model that learns the evolving structure of these financial networks and, based on historical data, accurately predicts asset returns for the next time step (t+1). By integrating both spatial information (asset interrelationships) and temporal dynamics (changes over time), the model outputs a vector of predicted returns for all 372 assets for the upcoming month. This approach captures not only the static features of individual assets but also how market conditions and interdependencies evolve, providing a robust forecasting tool for financial decision-making.

![image](https://github.com/user-attachments/assets/c2263ab3-b325-4a1d-b9ba-3412abd3e2b4)


Overview
Objective:
Predict the future (t+1) excess returns for all assets (372 nodes) based on historical dynamic graphs.
Method:
A graph neural network (GNN) is employed with the following components:
Graph Convolutional Layers (GCNConv): Extract node-level features from each graph.
GRU: Models the temporal dynamics across a sequence of graphs (historical time steps).
Fully Connected Layer: Maps the GRU output to the predicted return for each asset.
Data
Training Set:
features_train.pkl – Shape: (175 time steps, 372 assets, 94 features)
graph_train.pkl – Shape: (175 time steps, 372 nodes, 372 nodes)
Test Set:
features_test.pkl – Shape: (33 time steps, 372 assets, 93 features)
graph_test.pkl – Shape: (33 time steps, 372 nodes, 372 nodes)
Each training sample includes asset features (with the first column as the label for excess return) and a corresponding graph structure. The test set does not include labels.

Model Architecture
TGCN Model
Graph Convolutional Layers:
A list of GCNConv layers is applied over the historical time steps to extract spatial (graph) features for each asset.

GRU Layer:
The output from the GCN layers (collected over the historical window) is fed into a GRU layer to capture temporal dependencies.

Fully Connected Layer:
The final hidden state from the GRU is passed through a linear layer to predict the excess return for each asset.

Dropout:
Dropout is applied to the GCN outputs to prevent overfitting.

Training Details
Data Preprocessing:

Asset features are preprocessed using scaling (StandardScaler, MinMaxScaler, or RobustScaler) and optionally reduced via PCA.
Adjacency matrices are normalized if specified.
Dataset Splitting:
The data is split into training and validation sets based on time steps.

Training Loop:

Loss: Mean Squared Error (MSE) between predicted and actual returns.
Optimization: Adam optimizer with weight decay.
Learning Rate Scheduler: ReduceLROnPlateau is used to adjust the learning rate based on validation loss.
Early Stopping: Training stops if validation loss does not improve over a specified number of epochs.
Testing:
After training, the best model is loaded to generate predictions for the test set. The predicted returns for all assets are saved in a CSV file for submission.

Code Structure
Notebook:

DL_ap2938_Week4.ipynb - Colab – Contains the full implementation, including:
Data loading and preprocessing.
Definition of the AssetDataset and AssetBatch classes.
Implementation of the TGCNModel class.
Training, validation, and testing loops.
Generation of the submission CSV file.
Dependencies:

torch_geometric for graph neural network operations.
easydict, numpy, pandas, pickle, scipy, and sklearn for data processing.
torch and torch.nn for building and training the model.
Conclusion
This project demonstrates the application of graph neural networks combined with recurrent layers to capture both spatial and temporal dynamics in financial networks. By processing historical graphs representing asset interdependencies, the model predicts future asset returns, which can be evaluated using RMSE. All code, experiments, and results are provided in this repository.
