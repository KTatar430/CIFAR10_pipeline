"""Utility entry point for the CIFAR-10 pipeline.

Running this module will train a model and then evaluate it using the default
parameters and data paths defined in the existing scripts. This gives you a
"one command" way to exercise the full pipeline.

Usage::

    python main.py

The training script stores its best model in ``best_model.pth``, and the
evaluation step will read from that file.
"""

from train import train
from evaluate import evaluate


if __name__ == "__main__":
    # train and validate the model
    train()

    # evaluate the model on the test set
    evaluate()
