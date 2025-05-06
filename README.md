## **Hierarchical Spatio-Temporal Graph Learning Based on Metapath Aggregation for Emergency Supply Forecasting**
This repository is the official implementation of "Hierarchical Spatio-Temporal Graph Learning Based on Metapath Aggregation for Emergency Supply Forecasting" .
![3_model_sec.png](../_resources/3_model_sec.png)
## **Abstract**
Integrated Warehousing and Distribution Supply Networks (IWDSN), a typical form of logistics management, have shown their high efficiency and great applications in E-commerce. Efficient supply capacity prediction is crucial for logistics systems to maintain the delivery capacity to meet users' requirements. However, unforeseen events such as extreme weather and public health emergencies pose challenges in supply forecasting. Previous work mainly infers supply optimization based on the invariant topology of logistic networks, which may fail to capture the dynamic routing and distinct node effects reacting to emergencies. To address these challenges, the hierarchical relations among warehouses, sorting centers, and delivery stations in logistic networks are necessary to learn the diverse reactions. In this paper, we propose a hierarchical spatio-temporal graph learning model to predict the emergency supply capacity of IWDSN, based on the construction of micro and macro graphs. The micro graph shows transportation connectivity while the macro graph shows the geographical correlation. Specifically, it consists of three key components. (1) For micro graphs, a metapath aggregation strategy is designed to capture dynamic routing information on both route-view and event-view graphs. (2) For macro graphs, a bipartite graph learning approach to extract spatial representations. (3) For spatio-temporal feature fusion, the spatio-temporal joint forecasting module combines the temporal feature from the time-series encoder and the hierarchical spatial feature and predicts the future supply capacity. The extensive experiments on two real-world datasets from different cities demonstrate the effectiveness of our proposed model, which achieves state-of-the-art performance compared with advanced baselines.
## Requirements
Main package requirements:
- `CUDA == 11.7`
- `Python == 3.8.19`
- `PyTorch == 1.13.0`
- `DGL == 1.1.3`

## Quick Start
To train the HSTGL, run the following command in the directory `./`:
```Python
Python main.py --evtype=<event_type> --dataset=<dataset>
```
Explanation for the arguments:
- `evtype`:name for event types. `covid` and `weather` are available.
- `dataset`:name for datasets. `bj` and `sh` are available.