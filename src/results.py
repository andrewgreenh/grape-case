import models.simple.results as simple_results
from pathlib import Path

results_by_name = {
    'simple': simple_results,
}

name = 'simple'

model_directory = Path(__file__).parent.parent / 'results' / name
model_directory.mkdir(parents=True, exist_ok=True)
model_directory.resolve()

results_by_name[name].results(model_directory)
