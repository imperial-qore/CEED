import json
from .scheduler import Scheduler, ScheduleAction, time_wrapper
from .single_exit import SingleExitScheduler
from .adaee import AdaEE, AdaEEBuffer, AdaEEOrigin


__all__ = [
    'ScheduleAction', 'time_wrapper',
    'Scheduler',
    'SingleExitScheduler',
    'AdaEE', 'AdaEEBuffer', 'AdaEEOrigin',
    'init_scheduler'
]


def init_scheduler(model_profile, system_profile, scheduler_config, maxsize=None, max_memory_size=None,
                   log_path=None):
    if isinstance(scheduler_config, str):
        with open(scheduler_config, "r") as f:
            scheduler_config = json.load(f)
    scheduler_type = scheduler_config.pop('type')
    arrival_estimate_window = scheduler_config.pop('arrival_estimate_window', None)
    if scheduler_type == 'single_exit':
        return SingleExitScheduler(model_profile, system_profile, maxsize=maxsize, max_memory_size=max_memory_size,
                                   arrival_estimate_window=arrival_estimate_window, log_path=log_path,
                                   **scheduler_config)
    if scheduler_type == 'adaee':
        return AdaEE(model_profile, system_profile, maxsize=maxsize, max_memory_size=max_memory_size, log_path=log_path,
                     arrival_estimate_window=arrival_estimate_window, **scheduler_config)
    elif scheduler_type == 'adaee_origin':
        return AdaEEOrigin(model_profile, system_profile, maxsize=maxsize, log_path=log_path,
                           arrival_estimate_window=arrival_estimate_window, **scheduler_config)
    elif scheduler_type == 'adaee_buffer':
        return AdaEEBuffer(model_profile, system_profile, maxsize=maxsize, log_path=log_path,
                           arrival_estimate_window=arrival_estimate_window, **scheduler_config)
    else:
        raise ValueError(f"invalid scheduler type {scheduler_type}")
