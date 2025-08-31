# RRK Prune: Structural Pruning for Stable Diffusion Exploring Efficient Model Inference

# Quickstart
1. Setup python and pip, install requirements 
```bash
pip install -r requirements.txt
```
2. Generate labels in your data directory. Configure your directory in vars.py
```bash
python get_labels.py
```
3. Generate DeepCache data.
```bash
python run_deepcache.py
```