## **H^2DGL: Adaptive Metapath-based Dynamic Graph Learning for Supply Forecasting in Logistics System**
This repository is the official implementation of "H^2DGL: Adaptive Metapath-based Dynamic Graph Learning for Supply Forecasting in Logistics System" .
## **Abstract**
The advanced logistics systems are increasingly transitioning towards integrated warehousing and distribution supply networks (IWDSN), where accurately forecasting supply capacity is essential for maintaining delivery capabilities that meet user demands. However, existing research often overlooks the impact of dynamic changes in network topology, resulting in limitations in capturing dynamic routing and diverse node responses. These limitations become particularly pronounced in the context of external events such as pandemics, heavy rain, and promotions. To address the above limitations, we propose H^2DGL, a Hierarchical Heterogeneous Dynamic Graph Learning framework based on adaptive metapath aggregation, for forecasting supply capabilities in logistics systems. 
Specifically, H^2DGL comprises three main modules: (1) Hierarchical Heterogeneous Node Representation, where the micro graph captures dynamic routing information through adaptive meta-path aggregation from routing and event view graphs, and the macro graph extracts spatial representations using bipartite graph learning. (2) The Dynamic Graph Encoding module integrates macro and micro features from different snapshots to derive unified node representations. (3) The Spatio-temporal Joint Forecasting combines spatial features with temporal features from a time-series encoder to predict future supply capacity. Extensive experiments on two real-world datasets from different cities demonstrate that H^2DGL achieves state-of-the-art performance compared to advanced baseline models. 
## Requirements
Main package requirements:
- `CUDA == 11.7`
- `Python == 3.8.19`
- `PyTorch == 1.13.0`
- `DGL == 1.1.3`

## Quick Start
To train the H^2DGL, run the following command in the directory `./`:
```Python
Python main.py --evtype=<event_type> --dataset=<dataset>
```
Explanation for the arguments:
- `evtype`:name for event types. `covid` and `weather` are available.
- `dataset`:name for datasets. `bj` and `sh` are available.
