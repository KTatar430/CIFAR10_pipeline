# CIFAR-10 Training Pipeline

This simple Python project trains a convolutional neural network on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset using PyTorch. The repo contains routines for data loading, model definition, training with early stopping, and evaluation.

---

## 📁 Repository Structure

```
CIFAR10_pipeline/
├── data/
│   └── dataset.py         # data loader with augmentation
├── model/
│   └── cnn.py             # `SimpleCNN` model definition
├── train.py               # training script
├── evaluate.py            # evaluation script
├── requirements.txt       # Python dependencies
├── README.md              # this file
└── .gitignore             # files/directories to ignore
```

---

## 🛠 Setup

1. **Clone the repo**

   ```bash
   git clone <repo-url>
   cd CIFAR10_pipeline
   ```

2. **Create a virtual environment** (venv, conda, poetry, etc.) and activate it.

   ```powershell
   python -m venv venv          # Windows example
   .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > You may need a specific PyTorch build for your CUDA version; see https://pytorch.org/get-started/locally/ if `torch` fails to install.

---

## 🚀 Running the Project

Data are automatically downloaded into `./dataset` by default; the directory is created next to the `data/` package rather than inside it to avoid clutter.

### Train

```bash
python train.py
```

Hyperparameters (epochs, batch size, learning rate, patience) are hard‑coded at the top of the script; modify them directly if you need to change them.

### Evaluate

```bash
python evaluate.py
```

Both scripts will look for `best_model.pth` in the project root and `./dataset` for the CIFAR‑10 files. The evaluation output includes test loss/accuracy and a confusion matrix.


---

## 📦 Dependencies

The required packages are listed in `requirements.txt`. Minimum versions are specified for reproducibility:

- `torch` & `torchvision` (for the model and datasets)
- `pandas` & `scikit-learn` (for metrics / confusion matrix)
- `numpy`

Install with `pip install -r requirements.txt` after activating your environment.

---

## ⚙️ Customization

- Modify hyperparameters or add command‑line options in `train.py`/`evaluate.py`.
- The data loader (`get_dataloaders`) accepts a `root` path; calling code can override it.
- Set random seeds for reproducible results (not done by default).

---

## 🔒 Git Ignore

A `.gitignore` file is included to exclude large or generated files such as:

- `data/` directory (dataset downloads)
- model checkpoints (`*.pth`)
- Python virtual environments (`venv/`, etc.)
- `__pycache__/`, `.pytest_cache/`, etc.

---

Enjoy training!
