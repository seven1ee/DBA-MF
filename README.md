# DBA-MF

## Abstract
Molecular property prediction plays a crucial role in accelerating drug discovery, yet real-world data scarcity severely limits model generalization. Meta-learning provides an effective solution in molecular property prediction with limited labeled samples. In this paper, Dual-Branch Augmentation Multimodal Few-shot learning (DBA-MF) is proposed for molecular property prediction that integrates information from topological structures and geometric structures. In addition, a dual-branch augmentation module is designed that applies distinct augmentation strategies to the support set and the query set, respectively. Comparative experiments on public benchmarks demonstrate the effectiveness of DBA-MF, with performance surpassing state-of-the-art methods.
## Requirements

#### Download projects

Download the GitHub repo of this project onto your local server: 

```
git clone https://github.com/seven1ee/DBA-MF.git
```

#### Create environment

Create and activate virtual env using conda:

```
conda create -n "env_name" python=3.7
conda activate "env_name"
```
Install Pytorch and Torch_geometric:

```
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch_scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html
pip install torch_sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html
pip install torch_geometric==1.7.0
```
