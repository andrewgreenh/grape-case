from helpers import now
ms_until_stop = 25 * 60 * 1000
stop_at_ms = now() + ms_until_stop

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import models.simple.trainer as simple_trainer
import models.simple.split_trainer as simple_split_trainer
import models.cohen.trainer as cohen_trainer
import models.cohen.split_trainer as cohen_split_trainer
import models.marsden.trainer as marsden_trainer
import models.marsden.split_trainer as marsden_split_trainer
import models.wang.trainer as wang_trainer
import models.wang.split_trainer as wang_split_trainer
from pathlib import Path

trainers_by_name = {
    'simple': simple_trainer,
    'simple_split': simple_split_trainer,
    'cohen': cohen_trainer,
    'cohen_split': cohen_split_trainer,
    'marsden': marsden_trainer,
    'marsden_split': marsden_split_trainer,
    'wang': wang_trainer,
    'wang_split': wang_split_trainer,
}

name = 'wang'

model_directory = Path(__file__).parent.parent / 'results' / name
model_directory.mkdir(parents=True, exist_ok=True)
model_directory.resolve()

trainer = trainers_by_name[name].get_trainer(model_directory)

trainer.start_training(stop_at_ms)
