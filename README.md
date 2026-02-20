# CIFAR-10 Training Pipeline

This Python project trains a convolutional neural network on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset using PyTorch.

---

## Repository Structure

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

 **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

Data is automatically downloaded into `./dataset` by default;

### Run Full Pipeline (Training + Evaluation)

```bash
python main.py
```

This runs both training and evaluation sequentially in a single command. The training step saves the best model in `best_model.pth`, and the evaluation step loads it and tests on the test set.

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

## Dependencies

The required packages are listed in `requirements.txt`:

- `torch` & `torchvision` (for the model and datasets)
- `pandas` & `scikit-learn` (for metrics / confusion matrix)
- `numpy`

Install with `pip install -r requirements.txt` after activating your environment.

---
