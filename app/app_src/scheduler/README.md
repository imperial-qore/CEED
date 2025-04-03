# Example usage of scheduler
This is an example of how to use the scheduler module in a Python script.

## Initialize the scheduler
```python
from app.app_src.scheduler import init_scheduler

model_profile = "an example model profile in scheduler/example_configs"
system_profile = "an example system profile in scheduler/example_configs"
scheduler_config = "an example scheduler config in scheduler/<scheduler_type>/example_configs"

scheduler = init_scheduler(model_profile, system_profile, scheduler_config, maxsize=10)
```

## Get a scheduling decision
```python
action = scheduler.schedule()
```

## Update the scheduler if needed
```python
scheduler.update(update_kargs)
```