# CNN Image Classification (TensorFlow / Keras)

**What**: Full pipeline for CNN image classification using CIFAR-10 (default). Includes data preprocessing, model training, evaluation metrics, visualizations, and a Streamlit demo.

## Structure
See repo structure in README.

## Quick start

1. Create a Python environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train model:
bash run_train.sh

# Evaluate:

python src/evaluate.py --model_path outputs/models/best_model.h5


# Run Streamlit demo:

streamlit run src/app.py

# Dataset

data/custom/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/


# Notes


---

### `run_train.sh`
```bash
#!/usr/bin/env bash
python src/train.py --epochs 25 --batch_size 64 --use_augmentation True
