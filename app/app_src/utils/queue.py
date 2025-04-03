import time
from queue import Full
import multiprocessing
from multiprocessing.queues import Queue as mpQueue
from multiprocessing import Manager

import numpy as np


class FixedWindowProcess:
    def __init__(self, max_size):
        manager = Manager()
        self.max_size = max_size
        self.items = manager.list()
        self.lock = manager.Lock()

    def append(self, item):
        with self.lock:
            self.items.append(item)
            if len(self.items) > self.max_size:
                self.items.pop(0)

    def arrival_estimate(self):
        with self.lock:
            if len(self.items) <= 1:
                return 1
            difference = np.diff(self.items)
            return np.mean(difference)

    def __len__(self):
        with self.lock:
            return len(self.items)

    def __str__(self):
        with self.lock:
            return str(self.items)


class FiniteCapacityRegion:
    def __init__(self, maxsize: int = None, max_storage_size: int = None, window: int = None, ctx=None):
        assert window is not None or maxsize is not None, ValueError(
            "window must be provided if maxsize is not provided")

        if max_storage_size is not None:
            if maxsize is not None:
                maxsize = maxsize
            else:
                maxsize = 0  # 0 means infinite queue size
            self.memory_flag = multiprocessing.Value('b', True)
        else:
            max_storage_size = 0
            self.memory_flag = multiprocessing.Value('b', False)

        self._max_size = maxsize
        self._max_memory = max_storage_size

        self._current_size = multiprocessing.Value('i', 0)
        self._current_memory = multiprocessing.Value('i', 0)

        manager = Manager()
        self.checked_task_id = manager.list()

        if ctx is None:
            ctx = multiprocessing.get_context('spawn')
        self.ctx = ctx
        self.process_queue = mpQueue(ctx=ctx)
        self.send_queue = mpQueue(ctx=ctx)

        if window is None:
            window = maxsize
        self.arrival_window = FixedWindowProcess(window)

    def one_step_put(self, obj, is_send=False):
        self.pre_add(obj)
        if is_send:
            self.send_put_nowait(obj)
        else:
            self.put_nowait(obj)

    def pre_add(self, obj):
        """
        Pre-add task to the queue. This method is used to check whether the task can be added to the queue.
        Especially for network transmission. The region can check whether the task can be added to the queue before
        transmitting the task to the region.
        :param obj:
        :return:
        """
        self.arrival_window.append(time.perf_counter())
        assert isinstance(obj, dict), ValueError("obj must be a dictionary when max_storage_size is provided")
        assert 'task_id' in obj, ValueError("task_id must be provided in obj")
        task_id = obj['task_id']

        if self.memory_flag.value:
            assert 'data_size' in obj, ValueError("data_size must be provided in obj")
            size = obj['data_size']
            new_size = self._current_memory.value + size
            if new_size > self.max_memory:
                raise Full
            with self._current_memory.get_lock():
                self._current_memory.value = new_size
        else:
            if self._current_size.value + 1 > self._max_size:
                raise Full
            with self._current_size.get_lock():
                self._current_size.value += 1

        # can add task. add task_id to checked_task_id
        self.checked_task_id.append(task_id)

    def put(self, obj, block=True, timeout=None):
        assert isinstance(obj, dict), ValueError("obj must be a dictionary when max_storage_size is provided")
        assert 'task_id' in obj, ValueError("task_id must be provided in obj")
        task_id = obj['task_id']
        if task_id not in self.checked_task_id:
            raise ValueError(
                f"Task {task_id} is not pre-added to the queue. Please use pre_add method to check availability before adding task to the queue")
        self.checked_task_id.remove(task_id)
        self.process_queue.put(obj, block=block, timeout=timeout)

    def put_nowait(self, obj):
        return self.put(obj, block=False)

    def one_step_get(self, is_send=False):
        if is_send:
            obj = self.send_get_nowait()
        else:
            obj = self.get_nowait()
        self.leave(obj)
        return obj

    def get(self, block=True, timeout=None):
        item = self.process_queue.get(block=block, timeout=timeout)
        return item

    def get_nowait(self):
        return self.get(block=False)

    def send_put(self, obj, block=True, timeout=None):
        self.send_queue.put(obj, block=block, timeout=timeout)

    def send_put_nowait(self, obj):
        return self.send_put(obj, block=False)

    def send_get(self, block=True, timeout=None):
        item = self.send_queue.get(block=block, timeout=timeout)
        return item

    def send_get_nowait(self):
        return self.send_get(block=False)

    def leave(self, obj):
        if self.memory_flag.value:
            data_size = obj['data_size']
            with self._current_memory.get_lock():
                self._current_memory.value -= data_size
        else:
            with self._current_size.get_lock():
                self._current_size.value -= 1

    def change_queue_length(self, max_queue_length: int, max_memory: int = None):
        new_region = FiniteCapacityRegion(maxsize=max_queue_length, max_storage_size=max_memory, ctx=self.ctx)
        loss_task = []
        while not self.process_queue.empty():
            task = self.get_nowait()
            try:
                new_region.put_nowait(task)
            except Exception as e:
                loss_task.append(task)
        while not self.send_queue.empty():
            task = self.send_get_nowait()
            try:
                new_region.send_put_nowait(task)
            except Exception as e:
                loss_task.append(task)
        return new_region, loss_task

    def restart_queue(self):
        while not self.process_queue.empty():
            self.get_nowait()
        while not self.send_queue.empty():
            self.send_get_nowait()

    def estimate_arrival(self):
        return self.arrival_window.arrival_estimate()

    def empty(self):
        if self.process_queue.empty() and self.send_queue.empty():
            return True
        return False

    @property
    def max_memory(self):
        return self._max_memory

    @max_memory.setter
    def max_memory(self, value):
        self._max_memory = value

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, value):
        self._max_size = value

    @property
    def current_size(self):
        if self.memory_flag:
            return self._current_memory.value
        return self._current_size.value

    @property
    def current_queue(self):
        return self.process_queue.qsize()

    @property
    def current_memory(self):
        return self._current_memory.value