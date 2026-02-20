# CIFAR-10 Training Pipeline

This Python project trains a convolutional neural network on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset using PyTorch.

Key design decisions:

* **Data augmentation only on training set** – random crops and horizontal flips help generalization; validation and test sets remain unaltered to measure true model performance.
* **Fixed random seed** (42) ensures reproducible splits and training results across runs.
* **Simple, shallow CNN** chosen for clarity; three convolutional layers followed by two fully connected layers.
* **Single-entry point** (`main.py`) enables one‑line execution of the full pipeline while individual scripts support targeted runs.
* **Device-agnostic code** automatically uses CUDA if available, otherwise falls back to CPU.

Data augmentation:
Data augmentation is applied during training to artificially increase the diversity of the dataset. Random cropping and horizontal flipping should help the model generalize, reduce overfitting and improving performance on unseen test data.

Results:
Model achieves over 70% accuracy on the CIFAR-10 test set. Considering the simplicity of the CNN architecture and the fact that it was trained from scratch, this performance is reasonable. 

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
