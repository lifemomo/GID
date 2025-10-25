# GID: GNN-based Item Indexing for LLM-enhanced Recommendation

## Overview

This repository contains the implementation and reproduction code for **"GNN-based Item Indexing for LLM-enhanced Recommendation" (GID)**.  
GID proposes a graph neural network (GNN)-driven item indexing pipeline that generates semantically meaningful, compact item identifiers for LLM-based recommenders.  
The goal is to replace brittle heuristic IDs (e.g., random IDs, sequential position IDs) with structure-aware indices that (i) integrate collaborative signals, (ii) incorporate item attributes, and (iii) support scalable hierarchical clustering.

This codebase is adapted from the **OpenP5** platform for LLM-based recommendation, which provides data processing, prompt construction, training, and evaluation pipelines for sequence-aware recommenders. We extend, modify, and partially replace OpenP5 components to support our GNN-based indexing and analysis.  
If you use this repository, please also cite OpenP5 (see Citation).

> Paper: GNN-based Item Indexing for LLM-enhanced Recommendation  
> Paper status: under submission (ICDE 2026)  
> Code status: artifact for reproducibility

---

## What’s in this repo

- **Index construction (ours)**  
  - Builds an item co-occurrence graph and attribute-aware item features  
  - Trains a GNN under unsupervised objectives (contrastive / reconstruction / cosine)  
  - Performs hierarchical clustering to assign structured item IDs
- **LLM-based recommendation evaluation (from / extending OpenP5)**  
  - Prompted recommendation under "seen" / "unseen" prompt settings  
  - HR@K / NDCG@K evaluation for multiple datasets
- **Scripts for data preparation, training, and evaluation**  
  - End-to-end reproduction of the main tables in the paper for datasets such as Beauty

---

## Environment

We recommend creating a fresh environment before running any scripts.

If you are using `conda`, create and activate the environment with:
```bash
conda env create -f environment.yml
conda activate gid


If you are using plain pip, you can install:

pip install -r requirements.txt


Hardware notes:

Training the GNN and running evaluation is GPU-accelerated (PyTorch).

Inference/evaluation for the LLM-based recommender can also run on a single GPU.

CPU-only runs are possible for small subsets but may be slow.

Data

Some datasets used in the paper (e.g., Beauty, MovieLens, Last.fm, etc.) are public.

Download or generate the processed data:

sh generate_dataset.sh


This script prepares the interaction sequences and metadata.

Dataset-specific preprocessed files (e.g., item attributes, user sequences) are stored under directories like:

./Beauty/
./Movies/
./preprocessing/


If a dataset is not included directly in this repo due to licensing or size limits, we provide download instructions or scripts instead of raw data. Please follow the notes in the corresponding subdirectory README or script headers.

Running indexing + training
1. Build / update item indices using our GNN-based method
python src/train_gnn.py --config configs/beauty.yml
python src/build_index.py --config configs/beauty.yml


This will:

construct the item graph,

learn GNN embeddings using the chosen unsupervised loss,

perform hierarchical clustering to assign GID-style indices.

2. Run LLM-based recommendation with the generated indices

We provide command templates in command/ for training and in test_command/ for evaluation. For example:

cd command
sh beauty_seen.sh      # example: train under the "seen" prompt setting
cd ../test_command
sh beauty_seen.sh      # evaluate HR@K / NDCG@K


The produced metrics should match the main reported tables (HR@5/10, NDCG@5/10) within small variance.

Reproducing main results

A typical end-to-end pipeline for a dataset (e.g., Beauty) is:

# 0. (optional) prepare environment
conda env create -f environment.yml
conda activate gid

# 1. prepare data
sh generate_dataset.sh

# 2. learn GNN embeddings + build hierarchical indices
python src/train_gnn.py --config configs/beauty.yml
python src/build_index.py --config configs/beauty.yml

# 3. run LLM-based recommender with the learned indices
cd command
sh beauty_seen.sh
cd ../test_command
sh beauty_seen.sh


Evaluation scripts will report HR@K and NDCG@K under both "seen" and "unseen" prompt settings.

Notes on differences from OpenP5

Compared to the original OpenP5 codebase:

We replace / extend the item identifier generation step with a GNN-based hierarchical indexing module (GID).

We introduce unsupervised graph objectives (contrastive, link reconstruction, cosine similarity) for representation learning prior to indexing.

We incorporate item attributes (e.g., brand/category metadata) into the learned representations.

We add scripts for large-scale clustering and index assignment.

Other components (prompt templates, evaluation protocol, some data loaders, and baseline command scripts) are derived from or adapted from OpenP5.

License

This repository is released under the license specified in LICENSE.
Please also consult the original OpenP5 license and terms of use before redistributing derived components.

Citation

If you use this repository in academic work, please cite both our paper (when available) and OpenP5.

OpenP5:

@inproceedings{xu2024openp5,
  title     = {OpenP5: An Open-Source Platform for Developing, Training, and Evaluating LLM-based Recommender Systems},
  author    = {Shuyuan Xu and Wenyue Hua and Yongfeng Zhang},
  booktitle = {SIGIR},
  year      = {2024}
}


GID (this work):

@misc{gid2025,
  title         = {GNN-based Item Indexing for LLM-enhanced Recommendation},
  author        = {<Author list here>},
  note          = {Under submission},
  year          = {2025}
}
