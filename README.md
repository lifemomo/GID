# Reproducibility (Anonymous)

This repo reproduces the paper experiments based on OpenP5.

- Backbone: **T5-base**
- Tokenizer: **SentencePiece**
- Prompts: **all OpenP5 default templates**
- Other hyperparameters: **OpenP5 defaults**

## Install

Upgrade pip:
```bash
/root/autodl-tmp/venv_openp5/bin/python3 -m pip install --upgrade pip
````

PyTorch (CUDA 11.7):

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

Core deps:

```bash
pip install transformers==4.26.0 scikit-learn==1.1.2 tqdm==4.64.1 numpy==1.23.1 -i https://mirrors.aliyun.com/pypi/simple/
```

Data processing deps:

```bash
pip install torch-geometric pandas
```

## Run (in order)

### 1) Generate data

```bash
sh generate_dataset.sh
```

### 2) Train (Beauty example)

```bash
cd command
sh Beauty_collaborative.sh
```

### 3) Test (Beauty example)

```bash
cd test_command
sh Beauty_collaborative.sh
```

## Checkpoints

Download the checkpoint from Google Drive link(https://drive.google.com/drive/u/1/folders/1TYVg2PtiZfFFfEsR5MExCaJGlOOBd2c1), and put them into ./checkpoint folder. The evaluation command can be found in ./test_command folder. Run the command such as

cd ./test_command
sh Beauty_collaborative.sh
