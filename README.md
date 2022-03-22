# Awesome-Graph-Neural-Networks
This is the repo to collect latest materials of GNN, mainly focus on system context. Welcome to contribute!

## Talks
* [Intro to graph neural networks (ML Tech Talks)](https://www.youtube.com/watch?v=8owQBFAHw7E) - TensorFlow, Jun 18, 2021
* [An Introduction to Graph Neural Networks: Models and Applications](https://www.youtube.com/watch?v=zCEYiCxrL_0) - Microsoft, May 9, 2020
* [零基础多图详解图神经网络（GNN/GCN）【论文精读】](https://www.bilibili.com/video/BV1iT4y1d7zP/) - Mu Li, Nov.04 2021

## Blogs
* [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) - Distill, Sept.2 2021
* [Understanding Convolutions on Graphs](https://di]still.pub/2021/understanding-gnns/) - Distill, Sept.2 2021

## Survey
* [thunlp/GNNPapers](https://github.com/thunlp/GNNPapers)



## GNN Model
<div align="center"><h3>Graph Convolutional Network (GCN)</h3></div>

* [ICLR'17] [Semi-supervised classification with graphconvolutional networks](https://arxiv.org/pdf/1609.02907.pdf)
* [KDD'18] [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3219819.3219890) (PinSAGE)
* [NIPS'17] [Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) (GraphSAGE)

<div align="center"><h3>Graph Isomorphism Network (GIN)</h3></div>

* [ICLR'19] [How powerful are graph neural networks?](https://arxiv.org/abs/1810.00826)

<div align="center"><h3>Graph Recurrent Neural Network (GRN)</h3></div>

* To be added

<div align="center"><h3>Graph Attenttion Network (GAT)</h3></div>

* To be added


<div align="center"><h3>Unclassified</h3></div>

* None yet

## Network Embedding / Graph Embedding / Network Representation Learning

![](./doc/network_embedding_gnn.png)

With preserved origin information of vertexes and graph structure, **Network Embedding** focuses on representing each vertex as low-dimension vectors, while **GNN** is built within an end-to-end manner which takes in Graph Structure and Vertex/Edge Features to provides multiple machine learning task (e.g. edge prediction, graph classification). 

GNN is one of the methods to conduct Network Embedding, there're others including Matrix Factorization and Random Walk.

Network Embedding is one of the tasks that GNN is able to run through.
 
<div align="center"><h3>Matrix Factorization</h3></div>

* To be added

<div align="center"><h3>Random Walk</h3></div>

* [WWW'18] Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time [[Paper]](https://dl.acm.org/doi/abs/10.1145/3178876.3186183)

## System Design (Need to be further organized) 
* [PPoPP'22] Rethinking Graph Data Placement for Graph Neural Network Training on Multiple GPUs [[Paper]](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/37/POSTER-Rethinking-Graph-Data-Placement-for-Graph-Neural-Network-Training-on-Multiple)
* [PPoPP'22] Accelerating Quantized Graph Neural Networks via GPU Tensor Core [[Papers]](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/10/QGTC-Accelerating-Quantized-Graph-Neural-Networks-via-GPU-Tensor-Core)
* [PPoPP'22] Scaling Graph Traversal to 281 Trillion Edges with 40 Million Cores [[Paper]](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/5/Scaling-Graph-Traversal-to-281-Trillion-Edges-with-40-Million-Cores)
* [PPoPP'21] Understanding and Bridging the Gaps in CurrentGNN Performance Optimizations [[Paper]](https://dl.acm.org/doi/10.1145/3437801.3441585)
* [OSDI'21] GNNAdvisor: An Adaptive and Efficient Runtime System for GNN Acceleration on GPUs [[Paper]](https://www.usenix.org/conference/osdi21/presentation/wang-yuke) [[Presentation]](https://www.youtube.com/watch?v=K8Q7Dgko0Gs) [[Repo]](https://github.com/YukeWang96/OSDI21_AE) [[Chinese Blog from ZobinHuang]](https://www.zobinhuang.com:10443/sec_learning/Tech_Cloud_Network/Graph_Neural_Network_System_OSDI_21_GNNAdvisor/index.html)
* [OSDI'21] [Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads](https://www.usenix.org/conference/osdi21/presentation/thorpe)
* [OSDI'21] [Marius: Learning Massive Graph Embeddings on a Single Machine](https://www.usenix.org/conference/osdi21/presentation/mohoney)
* [OSDI'21] [P3: Distributed Deep Graph Learning at Scale](https://www.usenix.org/conference/osdi21/presentation/gandhi)
* [EuroSys'21] [Tripoline: generalized incremental graph processing via graph triangle inequality](https://dl.acm.org/doi/10.1145/3447786.3456226)
* [EuroSys'21] [DGCL: An Efficient Communication Library forDistributed GNN Training](https://dl.acm.org/doi/abs/10.1145/3447786.3456233)
* [EuroSys'21] [FlexGraph: a flexible and efficient distributed framework for GNN training](https://dl.acm.org/doi/10.1145/3447786.3456229)
* [EuroSys'21] [DZiG: Sparsity-Aware Incremental Processing ofStreaming Graphs](https://dl.acm.org/doi/10.1145/3447786.3456230)
* [EuroSys'21] [Accelerating graph sampling for graph machine learning using GPUs](https://dl.acm.org/doi/10.1145/3447786.3456244)
* [EuroSys'21] [Seastar: vertex-centric programming for graph neural networks](https://dl.acm.org/doi/10.1145/3447786.3456247)
* [EuroSys'21] [Tesseract: distributed, general graph pattern mining on evolving graphs](https://dl.acm.org/doi/10.1145/3447786.3456253)
* [NSDI'21] [GAIA: A System for Interactive Analysis on Distributed Graphs Using a High-Level Language](https://www.usenix.org/conference/nsdi21/presentation/qian-zhengping)
* [ATC'19] NeuGraph: Parallel Deep Neural Network Computation on Large Graphs [[Paper]](https://www.usenix.org/conference/atc19/presentation/ma) [[Presentation]](https://www.youtube.com/watch?v=avAiAy6VX4M) [[Chinese Blog from ZobinHuang]](https://www.zobinhuang.com:10443/sec_learning/Tech_Cloud_Network/Graph_Neural_Network_System_ATC_19_NeuGraph/index.html)

## Underlying System Design
* [OSDI'16] [Tensorflow: A system forlarge-scale machine learning](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)
* [OSDI'14] [Scaling Distributed Machine Learning  with the Parameter Server](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu)

## Software Library
* Deep Graph Library (DGL) [[Website]](https://www.dgl.ai/) [[Paper]](https://arxiv.org/abs/1909.01315) [[Repo]](https://github.com/dmlc/dgl)
* PyG (PyTorch Geometric) [[Repo]](https://github.com/pyg-team/pytorch_geometric) [[Paper]](https://arxiv.org/abs/1903.02428)
* SNAP (Stanford Network Analysis Platform) [[Website]](http://snap.stanford.edu/snap/index.html) [[Repo]](https://github.com/snap-stanford/snap)
