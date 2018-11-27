# Counting grapes on images

Contents:

- `data` Data set of pictures, where each grape has been marked with a coordinate in a respective txt file
- `src` python source code for model generation, training and result analysis
- `remote-results` results of the training of the vm
- `results` results of local training

Main file for training is `src/train.py`

## Concepts of implementation

Each model (subfolders of `src/models/`) has at least two different training configurations. One for the training with complete pictures in a reduced size and one for split pictures. Each training configuration implements certain methods that are needed for the training methodology (`strategy pattern`). These methods are injected into the `src/Trainer.py` class and are used during training. The `Trainer` class handles all algorithms regarding training, logging and persistence of results/history/model-weights.
