import models.simple.trainer as simple_trainer

trainers_by_name = {
    'simple': simple_trainer
}

name = 'simple'

trainer = trainers_by_name[name]

trainer.train()
