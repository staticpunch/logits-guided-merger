# Installation
```
pip install transformers datasets accelerate
```
# Run
```
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py configs/dryrun.yaml
```