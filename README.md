# Awesome-Graph-Neural-Networks
This is the repo to collect latest materials of GNN, mainly focus on system context. Welcome to contribute!

<div align="center"><h1>System Designs</h1></div>

## GNN System Design (Need to be further organized) 

### Distributed Sys Design
* [KDD'22] Distributed Hybrid CPU and GPU training for Graph Neural Networks on Billion-Scale Heterogeneous Graphs [[Paper]](https://assets.amazon.science/8e/01/c90c5c894771a5de766cbbba2b7e/distributed-hybrid-cpu-and-gpu-training-for-graph-neural-networks-on-billion-scale-heterogeneous-graphs.pdf)
* [MLSys'22] Sequential aggregation and rematerialization: Distributed full-batch training of graph neural networks on large graphs [[Paper]](https://proceedings.mlsys.org/paper/2022/file/5fd0b37cd7dbbb00f97ba6ce92bf5add-Paper.pdf)
* [KDD'21] Global Neighbor Sampling for Mixed CPU-GPU Training onGiant Graphs [[Paper]](https://arxiv.org/pdf/2106.06150.pdf)
* [OSDI'21] P3: Distributed Deep Graph Learning at Scale [[Paper]](https://www.usenix.org/conference/osdi21/presentation/gandhi)
* [OSDI'21] Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads [[Paper]](https://www.usenix.org/conference/osdi21/presentation/thorpe)
* [NSDI'21] GAIA: A System for Interactive Analysis on Distributed Graphs Using a High-Level Language [[Paper]](https://www.usenix.org/conference/nsdi21/presentation/qian-zhengping)
* [EuroSys'21] DGCL: An Efficient Communication Library for Distributed GNN Training [[Paper]](https://dl.acm.org/doi/abs/10.1145/3447786.3456233)
* [EuroSys'21] FlexGraph: a flexible and efficient distributed framework for GNN training [[Paper]](https://dl.acm.org/doi/10.1145/3447786.3456229)
* [IA3'20] Distdgl: distributed graph neural network training for billion-scale graphs [[Paper]](/https://arxiv.org/pdf/2010.05337.pdf)
* [SOSP'19] Knightking: a fast distributed graph random walk engine [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3341301.3359634)


### Single-machine Sys Design
* [VLDB'22] Marius++: Large-Scale Training of Graph Neural Networks on aSingle Machine [[Paper]](https://arxiv.org/pdf/2202.02365.pdf) [[Website]](https://marius-project.org/) [[Repo]](https://github.com/marius-team/marius)
* [MLSys'22] Accelerating training and inference of graph neural networks with fast sampling and pipelining [[Paper]](https://proceedings.mlsys.org/paper/2022/file/35f4a8d465e6e1edc05f3d8ab658c551-Paper.pdf)
* [PPoPP'22] Accelerating Quantized Graph Neural Networks via GPU Tensor Core [[Papers]](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/10/QGTC-Accelerating-Quantized-Graph-Neural-Networks-via-GPU-Tensor-Core)
* [PPoPP'21] Understanding and Bridging the Gaps in Current GNN Performance Optimizations [[Paper]](https://dl.acm.org/doi/10.1145/3437801.3441585)
* [OSDI'21] GNNAdvisor: An Adaptive and Efficient Runtime System for GNN Acceleration on GPUs [[Paper]](https://www.usenix.org/conference/osdi21/presentation/wang-yuke) [[Presentation]](https://www.youtube.com/watch?v=K8Q7Dgko0Gs) [[Repo]](https://github.com/YukeWang96/OSDI21_AE) [[Chinese Blog from ZobinHuang]](https://zobinhuang.github.io/sec_learning/Tech_Cloud_Network/Graph_Neural_Network_System_OSDI_21_GNNAdvisor/index.html)
* [OSDI'21] Marius: Learning Massive Graph Embeddings on a Single Machine [[Paper]](https://www.usenix.org/conference/osdi21/presentation/mohoney) [[Website]](https://marius-project.org/) [[Repo]](https://github.com/marius-team/marius)
* [VLDB'21] Large Graph Convolutional Network Training with GPU-Oriented Data Communication Architecture [[Paper]](https://arxiv.org/pdf/2103.03330.pdf)
* [EuroSys'21] Accelerating graph sampling for graph machine learning using GPUs [[Paper]](https://dl.acm.org/doi/10.1145/3447786.3456244) [[Repo]](https://github.com/plasma-umass/NextDoor)
* [EuroSys'21] Seastar: vertex-centric programming for graph neural networks [[Paper]](https://dl.acm.org/doi/10.1145/3447786.3456247)
* [MLSys'20] Improving the Accuracy, Scalability, and Performance of Graph Neural Networks with Roc [[Paper]](https://proceedings.mlsys.org/paper/2020/file/fe9fc289c3ff0af142b6d3bead98a923-Paper.pdf)
* [VLDB'20] G3: When Graph Neural Networks Meet Parallel GraphProcessing Systems on GPUs [[Paper]](https://dl.acm.org/doi/pdf/10.14778/3415478.3415482)
* [MLSys'19] Optimizing DNN computation with relaxed graph substitutions [[Paper]](https://proceedings.mlsys.org/paper/2019/file/b6d767d2f8ed5d21a44b0e5886680cb9-Paper.pdf)
* [ATC'19] NeuGraph: Parallel Deep Neural Network Computation on Large Graphs [[Paper]](https://www.usenix.org/conference/atc19/presentation/ma) [[Presentation]](https://www.youtube.com/watch?v=avAiAy6VX4M) [[Chinese Blog from ZobinHuang]](https://zobinhuang.github.io/sec_learning/Tech_Cloud_Network/Graph_Neural_Network_System_ATC_19_NeuGraph/index.html)
* [WWW'19] GraphVite: A High-Performance CPU-GPU Hybrid System forNode Embedding [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3308558.3313508)
* [Unknown] PyTorch-Direct: Enabling GPU Centric Data Access for Very Large Graph Neural Network Training with Irregular Accesses [[Paper]](https://arxiv.org/pdf/2101.07956.pdf)

## Traditional Graph Processing System Design
### Graph Analyse
* [PPoPP'22] Scaling Graph Traversal to 281 Trillion Edges with 40 Million Cores [[Paper]](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/5/Scaling-Graph-Traversal-to-281-Trillion-Edges-with-40-Million-Cores)
* [EuroSys'21] Tripoline: generalized incremental graph processing via graph triangle inequality [[Paper]](https://dl.acm.org/doi/10.1145/3447786.3456226)
* [EuroSys'21] DZiG: Sparsity-Aware Incremental Processing of Streaming Graphs [[Paper]](https://dl.acm.org/doi/10.1145/3447786.3456230)
* [EuroSys'20] Subway: Minimizing Data Transfer duringOut-of-GPU-Memory Graph Processing [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3342195.3387537)
* [ATC'19] SIMD-X: Programming and Processing of Graph Algorithms on GPUs [[Paper]](https://www.usenix.org/system/files/atc19-liu-hang.pdf) [[Repo]](https://github.com/asherliu/simd-x)
* [ASPLOS'18] Tigr: Transforming Irregular Graphs forGPU-Friendly Graph Processing [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3296957.3173180) [[Repo]](https://github.com/AutomataLab/Tigr)
* [PPoPP'16] Gunrock: A High-Performance Graph Processing Library on the GPU [[Paper]](https://dl.acm.org/doi/abs/10.1145/2851141.2851145) [[Repo]](https://github.com/gunrock/gunrock)
* [SC'15] Enterprise: Breadth-First Graph Traversal on GPUs  [[Paper]](https://dl.acm.org/doi/pdf/10.1145/2807591.2807594) [[Repo]](https://github.com/iHeartGraph/Enterprise)
* [HPDC'14] Cusha: vertex-centric graph pro-cessing on gpus [[Paper]](https://dl.acm.org/doi/pdf/10.1145/2600212.2600227) [[Repo]](https://github.com/farkhor/CuSha)
* [OSDI'12] PowerGraph: Distributed Graph-Parallel Computation on Natural Graphs [[Paper]](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/gonzalez)
* [SIGMOD'10] Pregel: A System for Large-Scale Graph Processing [[Paper]](https://dl.acm.org/doi/pdf/10.1145/1807167.1807184)

### Graph Mining
* [EuroSys'21] Tesseract: distributed, general graph pattern mining on evolving graphs [[Paper]](https://dl.acm.org/doi/10.1145/3447786.3456253)
* [VLDB'20] Pangolin: An Efficient and Flexible Graph Mining Systemon CPU and GPU [[Paper]](https://dl.acm.org/doi/pdf/10.14778/3389133.3389137)
* [SOSP'19] AutoMine: Harmonizing High-Level Abstraction and High Performance for Graph Mining [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3341301.3359633)
* [OSDI'18] ASAP: Fast, Approximate Graph  Pattern Mining at Scale [[Paper]](/https://www.usenix.org/system/files/osdi18-iyer.pdf)
* [SOSP'15] Arabesque: A System for Distributed Graph Mining [[Paper]](https://dl.acm.org/doi/pdf/10.1145/2815400.2815410)


## Open Source System
* Deep Graph Library (DGL) [[Website]](https://www.dgl.ai/) [[Paper]](https://arxiv.org/abs/1909.01315) [[Repo]](https://github.com/dmlc/dgl)
* PyG (PyTorch Geometric) [[Repo]](https://github.com/pyg-team/pytorch_geometric) [[Paper]](https://arxiv.org/abs/1903.02428)
* Cogdl [[Repo]](https://github.com/THUDM/cogdl) [[Paper]](https://arxiv.org/pdf/2103.00959.pdf)
* Graph-Learn [[Repo]](https://github.com/alibaba/graph-learn) [[Paper]](https://arxiv.org/pdf/1902.08730.pdf)
* PyTorch-BigGraph [[Repo]](https://github.com/facebookresearch/PyTorch-BigGraph) [[Paper]](https://mlsys.org/Conferences/2019/doc/2019/71.pdf)
* Gunrock [[Repo]](https://github.com/gunrock/gunrock) [[Paper]](https://dl.acm.org/doi/abs/10.1145/2851141.2851145)
* SNAP (Stanford Network Analysis Platform) [[Website]](http://snap.stanford.edu/snap/index.html) [[Repo]](https://github.com/snap-stanford/snap)

## Open Source Dataset
* Open Graph Benchmark [[Website]](https://ogb.stanford.edu/)
