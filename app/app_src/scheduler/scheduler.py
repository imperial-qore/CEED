import json
import os.path
from typing import Union, Dict
import time
from queue import Queue, Full
import multiprocessing

import numpy as np

from ..utils import Logger, FiniteCapacityRegion


def time_wrapper(func):
    # Time decorator for class method
    def wrapper(instance, *args, **kwargs):
        if instance.time_flag:
            start = time.perf_counter()
            result = func(instance, *args, **kwargs)
            instance._schedule_time = time.perf_counter() - start
            instance.mark_time()
        else:
            result = func(instance, *args, **kwargs)
        return result

    return wrapper


class ScheduleAction(dict):
    ALLOWED_KEYS = {"single_gate", "multi_gates", "skip_exit_threshold"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume = False
        self.transmit = None
        self.transmit_destination = None
        for k, v in kwargs.items():
            self.__setitem__(k, v)
        self._validate_keys()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._validate_keys()

    def __str__(self):
        attributes = ', '.join(f"{key}={self.__getitem__(key)}" for key in self.__iter__())
        return f"{self.__class__.__name__}({attributes})"

    def _validate_keys(self):
        # if not self.keys() <= self.ALLOWED_KEYS:
        #     raise KeyError(f"Only allowed keys are {', '.join(self.ALLOWED_KEYS)}.")

        if not (bool(self.get("single_gate", None) is None) ^ bool(self.get("multi_gates", None) is None)):
            raise KeyError("Dictionary must contain either 'single_gate' or 'multi_gates', but not both or none.")


class Scheduler:
    def __init__(self, model_profile: Union[str, Dict], system_profile: Union[str, Dict], maxsize: int = None,
                 max_memory_size: int = None, arrival_estimate_window: int = None, log_path: str = None, **kwargs):
        """
        the basic scheduler class
        :param model_profile: can be a path to model profile or a dictionary of model profile
        :param system_profile: path to system profile
        :param max_memory_size: maximum memory size of task queue
        :param maxsize: maximum size of task queue
        :param log_path: logging path
        :param kwargs:
        see below for kwargs:
        :key arrival_estimate_window: window size for estimating arrival rate
        :key time_flag: whether to record schedule time
        other keywords are scheduler specific
        """
        # loading model profile
        if isinstance(model_profile, str):
            if os.path.isdir(model_profile):
                model_profile = os.path.join(model_profile, 'config.json')
            model_profile = self._load_profile(model_profile)
        self.model_profile = model_profile

        # loading system profile
        if isinstance(system_profile, str):
            system_profile = self._load_profile(system_profile)
        self.system_profile = system_profile

        """the gates in the model profile include the final gate"""
        self._gates = self.model_profile.get('early_exits_gates', None)

        assert isinstance(self._gates, list), ("invalid early_exits_gates in model profile. "
                                               "Model profile should contain key early_exits_gates "
                                               "and its value should be a list")

        # initialize queue
        self.maxsize = maxsize
        self.max_memory_size = max_memory_size
        self._fcr = FiniteCapacityRegion(maxsize=maxsize, max_storage_size=max_memory_size,
                                         window=arrival_estimate_window)

        # initialize logger
        if log_path is None:
            log_path = ''
        self.logger = Logger(os.path.join(log_path, 'scheduler.log'), log_console=False)

        # set transmit destinations
        self.partition_layers = kwargs.get("partition_layers", None)
        self.transmit_destinations = kwargs.get("transmit_destinations", None)
        if self.transmit_destinations is not None and isinstance(self.transmit_destinations, str):
            self.transmit_destinations = [self.transmit_destinations]

        # check whether to record schedule time
        self.time_flag = kwargs.get('time_flag', False)
        self._schedule_time = None

        loss_prob = kwargs.get('loss_prob', 0)
        self.manager = multiprocessing.Manager()
        self._lock = self.manager.Lock()
        self._loss_prob = self.manager.Value('d', loss_prob)

    def _transmit_action(self):
        transmit = False
        select_destination = None
        partition_layers = None
        destinations = None
        if self.transmit_destinations is not None and len(self.transmit_destinations) > 0:
            destinations = list(self.transmit_destinations).copy()

        if self.partition_layers is not None and destinations:
            partition_layers = list(self.partition_layers)

        if destinations:
            transmit = True
            select_destination = destinations.pop(0)
        return transmit, partition_layers, select_destination

    def mark_time(self):
        self.logger.log("INFO", f"Schedule time: {self._schedule_time:.4f}")

    def status(self):
        return {'arrival_rate': float(self._fcr.estimate_arrival()), 'loss_prob': float(self.loss_prob)}

    @property
    def scheduler_type(self):
        return "abstract"

    @property
    def schedule_time(self):
        return self._schedule_time

    @staticmethod
    def _load_profile(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, 'r', encoding='utf8') as f:
            profile = json.load(f)
        return profile

    @property
    def gates_num(self):
        return len(self.model_profile['early_exits_gates'])

    @property
    def fc_region(self):
        return self._fcr

    @fc_region.setter
    def fc_region(self, fcr: FiniteCapacityRegion):
        self._fcr = fcr

    @time_wrapper
    def schedule(self, **kwargs) -> ScheduleAction:
        raise NotImplementedError("scheduler method must be implemented")

    def update(self, *args, **kwargs):
        pass

    def manuel_update(self, *args, **kwargs):
        print(f"updated: {kwargs}")
        pass

    def get_info(self):
        return "Abstract scheduler"

    def restart_queue(self):
        self._fcr.restart_queue()

    def synchronize_fcr(self, fcr: FiniteCapacityRegion):
        while not fcr.process_queue.empty():
            obj = fcr.get_nowait()
            self._fcr.pre_add(obj)
            self._fcr.put_nowait(obj)
        while not fcr.send_queue.empty():
            obj = fcr.send_get_nowait()
            self._fcr.pre_add(obj)
            self._fcr.send_put_nowait(obj)

    def q_get(self):
        return self._fcr.get()

    def q_get_nowait(self):
        return self._fcr.get_nowait()

    def q_put_nowait(self, task):
        self._fcr.put_nowait(task)

    def q_put(self, task):
        self._fcr.put(task)

    @property
    def q_size(self):
        return self._fcr.current_queue

    @property
    def empty(self):
        return self._fcr.empty()

    def get_task(self):
        return self._fcr.get()

    def change_queue_length(self, max_queue_length: int = None, max_memory: int = None):
        new_queue, loss_job = self._fcr.change_queue_length(max_queue_length, max_memory)
        self._fcr = new_queue
        return loss_job

    def send_q_get(self):
        return self._fcr.send_get()

    def send_q_get_nowait(self):
        return self._fcr.send_get_nowait()

    def send_q_put_nowait(self, task):
        self._fcr.send_put_nowait(task)

    def send_q_put(self, task):
        self._fcr.send_put(task)

    def pre_add(self, task):
        return self._fcr.pre_add(task)

    def leave(self, task):
        self._fcr.leave(task)

    def centralized_update(self, *args, **kwargs):
        pass

    @property
    def loss_prob(self):
        return self._loss_prob.value

    @loss_prob.setter
    def loss_prob(self, value):
        if isinstance(value, (int, float)):
            with self._lock:
                self._loss_prob.value = value
