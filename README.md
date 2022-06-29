# Awesome-Graph-Neural-Networks
This is the repo to collect latest materials of GNN, mainly focus on system context. Welcome to contribute!

<div align="center"><h1>System Designs</h1></div>

## GNN System Design (Need to be further organized) 

### Distributed Sys Design

* [PPoPP'22] Accelerating Quantized Graph Neural Networks via GPU Tensor Core [[Papers]](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/10/QGTC-Accelerating-Quantized-Graph-Neural-Networks-via-GPU-Tensor-Core)
* [PPoPP'21] Understanding and Bridging the Gaps in Current GNN Performance Optimizations [[Paper]](https://dl.acm.org/doi/10.1145/3437801.3441585)
* 
### Single-machine Sys Design

* [OSDI'21] GNNAdvisor: An Adaptive and Efficient Runtime System for GNN Acceleration on GPUs [[Paper]](https://www.usenix.org/conference/osdi21/presentation/wang-yuke) [[Presentation]](https://www.youtube.com/watch?v=K8Q7Dgko0Gs) [[Repo]](https://github.com/YukeWang96/OSDI21_AE) [[Chinese Blog from ZobinHuang]](https://zobinhuang.github.io/sec_learning/Tech_Cloud_Network/Graph_Neural_Network_System_OSDI_21_GNNAdvisor/index.html)
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
* [arXiv] [AliGraph: A Comprehensive Graph Neural NetworkPlatform](https://arxiv.org/pdf/1902.08730.pdf)
* [MLSys'20] [Improving the Accuracy, Scalability, and Performance of GraphNeural Networks with Roc](https://proceedings.mlsys.org/paper/2020/file/fe9fc289c3ff0af142b6d3bead98a923-Paper.pdf)
* [MLSys'19] [Optimizing DNN computation with relaxed graph substitutions](https://proceedings.mlsys.org/paper/2019/file/b6d767d2f8ed5d21a44b0e5886680cb9-Paper.pdf)
* [ATC'19] NeuGraph: Parallel Deep Neural Network Computation on Large Graphs [[Paper]](https://www.usenix.org/conference/atc19/presentation/ma) [[Presentation]](https://www.youtube.com/watch?v=avAiAy6VX4M) [[Chinese Blog from ZobinHuang]](https://zobinhuang.github.io/sec_learning/Tech_Cloud_Network/Graph_Neural_Network_System_ATC_19_NeuGraph/index.html)

## Traditional Graph Processing System Design
* [PPoPP'22] Scaling Graph Traversal to 281 Trillion Edges with 40 Million Cores [[Paper]](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/5/Scaling-Graph-Traversal-to-281-Trillion-Edges-with-40-Million-Cores)
* [HPDC'14] Cusha:  vertex-centric graph pro-cessing on gpus [[Paper]](https://dl.acm.org/doi/pdf/10.1145/2600212.2600227) [[Repo]](https://github.com/farkhor/CuSha)
* [SC'15] Enterprise: Breadth-First Graph Traversal on GPUs  [[Paper]](https://dl.acm.org/doi/pdf/10.1145/2807591.2807594) [[Repo]](https://github.com/iHeartGraph/Enterprise)
* [ATC'19] SIMD-X: Programming and Processing of Graph Algorithms on GPUs [[Paper]](https://www.usenix.org/system/files/atc19-liu-hang.pdf) [[Repo]](https://github.com/asherliu/simd-x)
* [ASPLOS'18] Tigr: Transforming Irregular Graphs forGPU-Friendly Graph Processing [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3296957.3173180) [[Repo]](https://github.com/AutomataLab/Tigr)
* [PPoPP'16] Gunrock: A High-Performance Graph Processing Library on the GPU [[Paper]](https://dl.acm.org/doi/abs/10.1145/2851141.2851145) [[Repo]](https://github.com/gunrock/gunrock)
* [OSDI'12] PowerGraph: Distributed Graph-Parallel Computation on Natural Graphs [[Paper]](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/gonzalez)
* [SIGMOD'10] Pregel: A System for Large-Scale Graph Processing [[Paper]](https://dl.acm.org/doi/pdf/10.1145/1807167.1807184)

## Open Source System
* Deep Graph Library (DGL) [[Website]](https://www.dgl.ai/) [[Paper]](https://arxiv.org/abs/1909.01315) [[Repo]](https://github.com/dmlc/dgl)
* PyG (PyTorch Geometric) [[Repo]](https://github.com/pyg-team/pytorch_geometric) [[Paper]](https://arxiv.org/abs/1903.02428)
* Cogdl [[Repo]](https://github.com/THUDM/cogdl) [[Paper]](https://arxiv.org/pdf/2103.00959.pdf)
* PyTorch-BigGraph [[Repo]](https://github.com/facebookresearch/PyTorch-BigGraph) [[Paper]](https://mlsys.org/Conferences/2019/doc/2019/71.pdf)
* Gunrock [[Repo]](https://github.com/gunrock/gunrock) [[Paper]](https://dl.acm.org/doi/abs/10.1145/2851141.2851145)
* SNAP (Stanford Network Analysis Platform) [[Website]](http://snap.stanford.edu/snap/index.html) [[Repo]](https://github.com/snap-stanford/snap)

## Open Source Dataset
* Open Graph Benchmark [[Website]](https://ogb.stanford.edu/)
