from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import models.simple.trainer as simple_trainer
import models.simple.split_trainer as simple_split_trainer
import models.cohen.trainer as cohen_trainer
import models.cohen.split_trainer as cohen_split_trainer
import models.marsden.trainer as marsden_trainer
import models.marsden_600.trainer as marsden_600_trainer
import models.marsden.split_trainer as marsden_split_trainer
import models.wang.trainer as wang_trainer
import models.wang.split_trainer as wang_split_trainer

from pathlib import Path
import math

from helpers import now, str2bool

trainers_by_name = {
    'simple': simple_trainer,
    'simple_split': simple_split_trainer,
    'cohen': cohen_trainer,
    'cohen_split': cohen_split_trainer,
    'marsden': marsden_trainer,
    'marsden_600': marsden_600_trainer,
    'marsden_split': marsden_split_trainer,
    'wang': wang_trainer,
    'wang_split': wang_split_trainer,
}


import argparse

parser = argparse.ArgumentParser(
    description='Train different models for counting berries on images.')

parser.add_argument('--training-time', default=99*1000*60*60, type=int,  metavar='ms',
                    help='Available training time in ms. Model will check this time after each epoch')
parser.add_argument('--model-name', default='simple', choices=trainers_by_name.keys(),
                    help='Model that should be trained.')
parser.add_argument('--persistence-directory', default='./results', type=str,
                    help='Directory where model results and caches are stored.')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of epochs to train')
parser.add_argument('--multi-processing', default=False, type=str2bool,
                    help="Should multithreading be used for data augmentation")

args = parser.parse_args()

print('Arguments:', args)

ms_until_stop = args.training_time
model_name = args.model_name
persistence_directory = args.persistence_directory
epochs = args.epochs
multi_processing = args.multi_processing


stop_at_ms = now() + ms_until_stop

model_directory = Path(persistence_directory) / model_name
model_directory.mkdir(parents=True, exist_ok=True)
model_directory = model_directory.resolve()

print(model_directory)

trainer = trainers_by_name[model_name].get_trainer(model_directory)

trainer.start_training(stop_at_ms, epochs, multi_processing)
