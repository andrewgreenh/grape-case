from helpers import now
ms_until_stop = 25 * 60 * 1000
stop_at_ms = now() + ms_until_stop

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import models.simple.trainer as simple_trainer
from pathlib import Path

trainers_by_name = {
    'simple': simple_trainer
}

name = 'simple'

model_directory = Path(__file__).parent.parent / 'results' / name
model_directory.mkdir(parents=True, exist_ok=True)
model_directory.resolve()

trainer = trainers_by_name[name].get_trainer(model_directory)

trainer.start_training(stop_at_ms)
