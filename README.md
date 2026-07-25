# SpaHDSRL

## 1. Introduction

SpaHDSRL is a hierarchical dual-graph self-supervised representation learning framework for spatial multi-omics integration. It aims to learn a low-dimensional embedding from the spatially resolved multi-omics data, supporting downstream biological analyses such as marker gene discovery, functional enrichment, and cell-cell communication analysis.

### Repository structure

```text
├── Data
├── SpaHDSRL
│   ├── process.py
│   ├── model.py
│   └── crossLayer.py
├── Tutorial
├── LICENSE
└── README.md
```

- `Data/`: datasets required for reproduction.

- `SpaHDSRL/`: source code of the SpaHDSRL algorithm.
  - `process.py`: overall training pipeline.
  - `model.py`: main model framework.
  - `crossLayer.py`: model fusion module.

- `Tutorial/`: five demo files for reproducing the workflow.

## 2. Environment setup

We recommend using Conda to create the environment for SpaHDSRL:

```
conda create -n SpaHDSRL python=3.12 pandas numpy scanpy matplotlib umap-learn scikit-learn seaborn networkx gudhi anndata cmcrameri louvain leidenalg
conda activate SpaHDSRL
pip install torch torchvision torchaudio torch-geometric notebook 
```

For Jupyter Notebook usage, please also install and register the kernel:

```
pip install ipykernel
python -m ipykernel install --user --name=SpaHDSRL --display-name="SpaHDSRL"
```

## 3. Tutorials

Tutorial notebooks are provided to help users understand the overall workflow of SpaHDSRL, reproduce the main analysis procedures, and apply SpaHDSRL to their own paired spatial multi-omics datasets. Here, we present tutorials for two simulated datasets and three real datasets. The data required for running these tutorials are available in the `data/` folder.

## 4. Contact information

If you have any questions, please contact lixiang@stu.cpu.edu.cn.
