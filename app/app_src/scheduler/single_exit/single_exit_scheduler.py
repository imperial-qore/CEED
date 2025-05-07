import os.path
import queue
import time
from typing import List
import pickle
import socket
from multiprocessing import Manager

import numpy as np

from .model_base_utils import ModelBaseOptimisation
from ..scheduler import Scheduler, ScheduleAction, time_wrapper
from .utils.branch_n_bound import KnapSackBranchNBound
from .utils.utils import copy_mp_list
from ...utils import get_machine_type


class BaseSingleExitScheduler(Scheduler):
    def __init__(self, model_profile, system_profile, maxsize=None, max_memory_size=None,
                 arrival_estimate_window: int = None, log_path=None, **kwargs):
        super(BaseSingleExitScheduler, self).__init__(model_profile,
                                                      system_profile,
                                                      maxsize,
                                                      max_memory_size,
                                                      arrival_estimate_window,
                                                      log_path=log_path,
                                                      **kwargs)
        self.manager = Manager()
        self._gates_prob = self.manager.list([0] * len(self._gates))
        self._gates_prob_lock = self.manager.Lock()
        self._mode_name = 'base'

    @property
    def scheduler_type(self):
        return "single_exit"

    @property
    def mode(self):
        return self._mode_name

    @property
    def gates_prob(self):
        return self._gates_prob

    @time_wrapper
    def schedule(self, **kwargs):
        transmit, partition_layers, select_destination = self._transmit_action()
        exit_gate = np.random.choice(self.gates_num, p=self.gates_prob)
        action = ScheduleAction(single_gate=exit_gate,
                                transmit=transmit,
                                partition_layers=partition_layers,
                                transmit_destination=select_destination)
        self.logger.log("INFO", f"Exit Gate: {exit_gate}")
        return action

    def get_info(self):
        info = [
            "Scheduler: SingleExitScheduler",
            "Mode: {}".format(self.mode),
            "Gates probability: {}".format(self.gates_prob),
        ]
        return "\n".join(info)


class RandomMode(BaseSingleExitScheduler):
    """
    Random mode scheduler
    The scheduler randomly choose one gate to process the task
    """

    def __init__(self, model_profile, system_profile, maxsize, log_path=None, **kwargs):
        super(RandomMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path, **kwargs)
        self._mode_name = 'RandomMode'

        self._gates_prob = self.manager.list([1 / len(self._gates)] * len(self._gates))


class SingleMode(BaseSingleExitScheduler):
    """
    Single mode scheduler
    The scheduler only use one gate to process all the tasks
    """

    def __init__(self, model_profile, system_profile, maxsize, log_path=None, gate_id: int = None,
                 gate_layer: int = None, **kwargs):
        super(SingleMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path, **kwargs)
        self._mode_name = 'SingleMode'
        assert gate_id is not None or gate_layer is not None, "gate_id or gate_layer must be specified"

        self._gates_prob = self.manager.list([0] * len(self._gates))
        if gate_id is not None:
            assert gate_id < len(self._gates), f"gate_id must be less than {len(self._gates)}"
            gate = gate_id
        else:
            assert gate_layer in self._gates, "gate_layer must be in early_exits_gates"
            gate = self._gates.index(gate_layer)
        with self._gates_prob_lock:
            self._gates_prob[gate] = 1


class SpecifyMode(BaseSingleExitScheduler):
    def __init__(self, model_profile, system_profile, maxsize, log_path=None, gates_prob: List[int] = None, **kwargs):
        super(SpecifyMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path, **kwargs)
        self._mode_name = 'SpecificMode'
        assert len(gates_prob) == len(self._gates), f"gates_prob length must be {len(self._gates)}"
        if sum(gates_prob) != 1:
            gates_prob = [i / sum(gates_prob) for i in gates_prob]
        self._gates_prob = self.manager.list(gates_prob)


class ReactiveMode(BaseSingleExitScheduler):
    """
    Reactive mode scheduler
    The scheduler reduce one gate when the queue is full and increase one gate when the queue is empty
    """

    def __init__(self, model_profile, system_profile, maxsize, log_path=None, **kwargs):
        super(ReactiveMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path, **kwargs)
        self._mode_name = 'ReactiveMode'
        self._gates_prob = self.manager.list([0] * len(self._gates))
        with self._gates_prob_lock:
            self._gates_prob[-1] = 1

    def update_full(self):
        idx = self._gates_prob.index(1)
        with self._gates_prob_lock:
            self._gates_prob[idx] = 0
        idx_down = max(idx - 1, 0)
        with self._gates_prob_lock:
            self._gates_prob[idx_down] = 1

    def update_empty(self):
        idx = self._gates_prob.index(1)
        with self._gates_prob_lock:
            self._gates_prob[idx] = 0
        idx_up = min(idx + 1, len(self._gates_prob) - 1)
        with self._gates_prob_lock:
            self._gates_prob[idx_up] = 1

    def q_get(self):
        task = self._fcr.get()
        if self.empty:
            self.update_empty()
        return task

    def q_get_nowait(self):
        try:
            task = self._fcr.get_nowait()
        except queue.Empty:
            self.update_empty()
            raise queue.Empty
        return task

    def pre_add(self, task):
        try:
            super(ReactiveMode, self).pre_add(task)
        except queue.Full:
            self.update_full()
            raise queue.Full


class ModelBaseMode(BaseSingleExitScheduler):
    """
    Model based mode scheduler
    Based on G.Casale "Scheduling Inputs in Early Exit Neural Networks" model based scheduler
    """

    def __init__(self, model_profile, system_profile, maxsize, log_path=None, loss_ratio=1, num_sample_p=100,
                 update_threshold=5, **kwargs):
        super(ModelBaseMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path, **kwargs)
        self.num_sample_p = num_sample_p
        self.loss_ratio = loss_ratio
        self.update_threshold = update_threshold

        # initialise the gates probability
        with self._gates_prob_lock:
            self._gates_prob[-1] = 1

        self.exit_gates = self.model_profile['early_exits_gates']
        self.gates_process_time = list()
        self.gates_accuracy = list()
        for i, g in enumerate(self.exit_gates):
            process_time_info = self.system_profile.get(f"execution_time_gate_{i}_layer_{g}", None)
            if process_time_info is None:
                raise ValueError("model profile and system profile not match")
            self.gates_process_time.append(process_time_info['mean'])
            self.gates_accuracy.append(self.model_profile['accuracy'][f"gate_{i}"])
        self.estimate_arrival_rate = 0

        lookup_table_path = kwargs.get('lookup_table', None)

        if lookup_table_path is None:
            self.is_online = True
            self._mode_name = "ModelBaseMode"
            self._lp_solver = ModelBaseOptimisation(
                self.gates_process_time, self._fcr.max_size, self.num_sample_p, self.loss_ratio, **kwargs
            )
        else:
            self.is_online = False
            self._load_lookup_table(lookup_table_path)
            self._mode_name = "ModelBaseLookUpMode"

        self._record_gate_prob = kwargs.get('record_gate_prob', False) or kwargs.get('record', False)

    def _load_lookup_table(self, lookup_table_path):
        if lookup_table_path is None:
            raise ValueError("lookup_arrival_path is not provided")
        hostname = socket.gethostname()
        machine_type = get_machine_type(hostname)
        lookup_table_path = lookup_table_path.format(machine_type=machine_type, machine_name=hostname)
        if os.path.exists(lookup_table_path):
            with open(lookup_table_path, 'rb') as f:
                self._lookup_table = pickle.load(f)
        else:
            raise FileExistsError(f"lookup_arrival_path {lookup_table_path} not exists")
        self._lookup_arrivals = np.array(list(self._lookup_table.keys()))

    def get_info(self):
        info = [
            "Scheduler: SingleExitScheduler",
            "Mode: {}".format(self.mode),
            "Actual arrival: {}".format(self._fcr.estimate_arrival()),
            "Estimate arrival: {}".format(self.estimate_arrival_rate),
            "Gates probability: {}".format(self.gates_prob),
        ]
        return "\n".join(info)

    def _update_gate_prob_with_lookup_arrival(self, arrival_rate):
        estimate_arrival_rate = self._lookup_arrivals[
            np.argmin(np.abs(self._lookup_arrivals - arrival_rate))]
        # make sure we need to update the gates probability

        if estimate_arrival_rate != self.estimate_arrival_rate:
            self.estimate_arrival_rate = estimate_arrival_rate
            copy_mp_list(self._gates_prob, self._lookup_table[self.estimate_arrival_rate], self._gates_prob_lock)

    def _update_gate_prob_with_lp(self, arrival_rate):
        arrival_rate = 1 / arrival_rate
        gates_prob = self._lp_solver.optimisation(self.gates_process_time,
                                                  self.gates_accuracy,
                                                  arrival_rate)
        copy_mp_list(self._gates_prob, gates_prob, self._gates_prob_lock)

    def _update_criterion(self, arrival_rate, current_est_arrival):
        return np.abs(1 / arrival_rate - 1 / current_est_arrival) > self.update_threshold

    @time_wrapper
    def schedule(self, **kwargs):
        # arrival rate here is the inter-arrival time
        arrival_rate = self._fcr.estimate_arrival()
        current_est_arrival = self.estimate_arrival_rate if self.estimate_arrival_rate != 0 else np.inf

        # update the gates probability if the arrival rate is changed
        if self._update_criterion(arrival_rate, current_est_arrival):
            if self.is_online:
                self._update_gate_prob_with_lp(arrival_rate)
            else:
                self._update_gate_prob_with_lookup_arrival(arrival_rate)

        exit_gate = np.random.choice(self.gates_num, p=self.gates_prob)
        transmit, partition_layers, select_destination = self._transmit_action()
        action = ScheduleAction(single_gate=exit_gate,
                                transmit=transmit,
                                partition_layers=partition_layers,
                                transmit_destination=select_destination
                                )
        self.logger.log("INFO", f"Exit Gate: {exit_gate}")

        if self._record_gate_prob:
            self.logger.log("INFO", f"Arrival est: {arrival_rate}, Gate prob: {self.gates_prob}")
        return action


class ConfidenceModelBaseMode(ModelBaseMode):
    def __init__(self, model_profile, system_profile, maxsize, log_path=None, loss_ratio=1, num_sample_p=100,
                 update_threshold=5, update_threshold_conf=0.1, **kwargs):
        super(ConfidenceModelBaseMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path,
                                                      loss_ratio=loss_ratio, num_sample_p=num_sample_p,
                                                      update_threshold=update_threshold, **kwargs)
        self.update_threshold_conf = update_threshold_conf
        self._mode_name = 'ConfidenceModelBaseMode'

        self.confidence_window = kwargs.get('confidence_window', 20)
        self.gates_confidence = self.manager.list()
        for i, g in enumerate(self.exit_gates):
            with self._gates_prob_lock:
                self.gates_confidence.append(self.model_profile['confidence'][f"gate_{i}"])

        self._lp_solver = ModelBaseOptimisation(
            self.gates_process_time, self._fcr.max_size, self.num_sample_p, self.loss_ratio, **kwargs
        )

        self._conf_need_update = False
        self._record_gate_prob = kwargs.get('record_gate_prob', False) or kwargs.get('record', False)
        self._record_confidence = kwargs.get('record_confidence', False) or kwargs.get('record', False)

    def get_info(self):
        info = [
            "Scheduler: SingleExitScheduler",
            "Mode: {}".format(self.mode),
            "Actual arrival: {}".format(self._fcr.estimate_arrival()),
            "Gates confidence: {}".format(self.gates_confidence),
        ]
        return "\n".join(info)

    def update(self, action, confidences=None, **kwargs):
        """
        :param action:
        :param confidences: confidence is a single number in single exit mode
        :return:
        """
        if confidences:
            gate = action['single_gate']
            with self._gates_prob_lock:
                previous_confidence = self.gates_confidence[gate]
                self.gates_confidence[gate] = ((self.confidence_window - 1) * self.gates_confidence[
                    gate] + confidences) / self.confidence_window
                self._conf_need_update = np.abs(
                    previous_confidence - self.gates_confidence[gate]) > self.update_threshold_conf

    def _update_gate_prob_with_lp(self, arrival_rate):
        arrival_rate = 1 / arrival_rate
        gates_prob = self._lp_solver.optimisation(self.gates_process_time,
                                                  self.gates_confidence,
                                                  arrival_rate)
        copy_mp_list(self._gates_prob, gates_prob, self._gates_prob_lock)

    def _update_criterion(self, arrival_rate, current_est_arrival):
        return np.abs(1 / arrival_rate - 1 / current_est_arrival) > self.update_threshold or self._conf_need_update


class RateBaseMode(BaseSingleExitScheduler):
    def __init__(self, model_profile, system_profile, maxsize, log_path=None, **kwargs):
        super(RateBaseMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path, **kwargs)
        lookup_arrival_path = kwargs.get('lookup_arrival', '')
        if os.path.exists(lookup_arrival_path):
            self._lookup_arrival = np.load(lookup_arrival_path)
            self._mode_name = 'RateBaseLookUpMode'
        else:
            self._lookup_arrival = None
            self._mode_name = 'RateBaseMode'

        lookup_table = kwargs.get('lookup_table', False)
        self._lookup_table = None
        if lookup_table:
            lookup_table_path = os.path.join(os.path.dirname(system_profile), 'rate_based_lookup_table.pkl')
            with open(lookup_table_path, 'rb') as f:
                self._lookup_table = pickle.load(f)
            if isinstance(self._lookup_table, list):
                potential_table = [abs(i[0] - self._fcr.max_size) for i in self._lookup_table]
                select_table = np.argmin(potential_table)
                self._lookup_table = self._lookup_table[select_table][1]

        self.exit_gates = self.model_profile['early_exits_gates']
        self.gates_process_time = list()
        self.gates_accuracy = list()
        for i, g in enumerate(self.exit_gates):
            process_time_info = self.system_profile.get(f"execution_time_gate_{i}_layer_{g}", None)
            if process_time_info is None:
                raise ValueError("model profile and system profile not match")
            self.gates_process_time.append(process_time_info['mean'])
            self.gates_accuracy.append(self.model_profile['accuracy'][f"gate_{i}"])

        self._record_gate_prob = kwargs.get('record_gate_prob', False) or kwargs.get('record', False)

    def get_info(self):
        info = [
            "Scheduler: SingleExitScheduler",
            "Mode: {}".format(self.mode),
            "Actual arrival: {}".format(self._fcr.estimate_arrival()),
            "Gates probability: {}".format(self.gates_prob),
        ]
        return "\n".join(info)

    def select_from_table(self):
        arrival_rate = self._fcr.estimate_arrival()
        n_job = self._fcr.current_queue + 1
        closest_value = min(self._lookup_table.keys(), key=lambda x: abs(x - arrival_rate))
        close_arrival_job = self._lookup_table[closest_value]
        closest_job = min(close_arrival_job.keys(), key=lambda x: abs(x - n_job))
        num_exit_per_gate = close_arrival_job[closest_job]
        return num_exit_per_gate

    @time_wrapper
    def schedule(self, **kwargs):
        if self._lookup_table is not None:
            num_exit_per_gate = self.select_from_table()
        else:
            num_exit_per_gate = self.knapsack_bnb()
        copy_mp_list(self._gates_prob, self.convert_to_gates_prob(num_exit_per_gate), self._gates_prob_lock)
        exit_gate = np.random.choice(self.gates_num, p=self.gates_prob)
        transmit, partition_layers, select_destination = self._transmit_action()
        action = ScheduleAction(single_gate=exit_gate,
                                transmit=transmit,
                                partition_layers=partition_layers,
                                transmit_destination=select_destination
                                )
        self.logger.log("INFO", f"Exit Gate: {exit_gate}")
        if self._record_gate_prob:
            self.logger.log("INFO", f"Gate prob: {self.gates_prob}")
        return action

    def time_budget(self):
        if self._lookup_arrival is not None:
            arrival_rate = self._lookup_arrival[
                np.argmin(np.abs(self._lookup_arrival - self._fcr.estimate_arrival()))]
        else:
            arrival_rate = self._fcr.estimate_arrival()
        budge = arrival_rate * (self._fcr.max_size - self._fcr.current_queue)
        return budge

    def knapsack_bnb(self):
        n_job = self._fcr.current_queue + 1  # k
        budget_constrain = self.time_budget() - n_job * self.gates_process_time[0]  # W
        items = [(acc, pt) for acc, pt in zip(self.gates_accuracy[1:], self.gates_process_time[1:])]
        _, num_exit_per_gate = KnapSackBranchNBound.fit(budget_constrain, items, n_job)
        if num_exit_per_gate is None:
            num_exit_per_gate = np.zeros(len(self.gates_process_time), dtype=int)
            num_exit_per_gate[0] = n_job
        else:
            num_exit_per_gate = [n_job - int(sum(num_exit_per_gate))] + num_exit_per_gate
            num_exit_per_gate = np.array(num_exit_per_gate, dtype=int)

        return num_exit_per_gate

    @staticmethod
    def convert_to_gates_prob(num_exit_per_gate):
        n_jobs = np.sum(num_exit_per_gate)
        gates_prob = 1 / n_jobs * num_exit_per_gate
        gates_prob[0] = 1 - np.sum(gates_prob[1:])
        return gates_prob


class ConfidenceRateBaseMode(RateBaseMode):
    def __init__(self, model_profile, system_profile, maxsize, log_path=None, **kwargs):
        super(ConfidenceRateBaseMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path,
                                                     **kwargs)
        lookup_arrival_path = kwargs.get('lookup_arrival', '')
        if os.path.exists(lookup_arrival_path):
            self._lookup_arrival = np.load(lookup_arrival_path)
            self._mode_name = 'ConfidenceRateBaseLookUpMode'
        else:
            self._lookup_arrival = None
            self._mode_name = 'ConfidenceRateBaseMode'

        self.confidence_window = kwargs.get('confidence_window', 20)
        self.gates_confidence = self.manager.list()
        for i, g in enumerate(self.exit_gates):
            with self._gates_prob_lock:
                self.gates_confidence.append(self.model_profile['confidence'][f"gate_{i}"])

        self._record_gate_prob = kwargs.get('record_gate_prob', False) or kwargs.get('record', False)
        self._record_confidence = kwargs.get('record_confidence', False) or kwargs.get('record', False)

    def get_info(self):
        info = [
            "Scheduler: SingleExitScheduler",
            "Mode: {}".format(self.mode),
            "Actual arrival: {}".format(self._fcr.estimate_arrival()),
            "Gates confidence: {}".format(self.gates_confidence),
        ]
        return "\n".join(info)

    def knapsack_bnb(self):
        n_job = self._fcr.current_queue + 1  # k
        budget_constrain = self.time_budget() - n_job * self.gates_process_time[0]  # W
        items = [(acc, pt) for acc, pt in zip(self.gates_confidence[1:], self.gates_process_time[1:])]
        _, num_exit_per_gate = KnapSackBranchNBound.fit(budget_constrain, items, n_job)
        if num_exit_per_gate is None:
            num_exit_per_gate = np.zeros(len(self.gates_process_time), dtype=int)
            num_exit_per_gate[0] = n_job
        else:
            num_exit_per_gate = [n_job - int(sum(num_exit_per_gate))] + num_exit_per_gate
            num_exit_per_gate = np.array(num_exit_per_gate, dtype=int)

        return num_exit_per_gate

    def update(self, action, confidences=None, **kwargs):
        """
        :param action:
        :param confidences: confidence is a single number in single exit mode
        :return:
        """
        if confidences:
            gate = action['single_gate']
            with self._gates_prob_lock:
                self.gates_confidence[gate] = ((self.confidence_window - 1) * self.gates_confidence[
                    gate] + confidences) / self.confidence_window

    @time_wrapper
    def schedule(self, **kwargs):
        num_exit_per_gate = self.knapsack_bnb()
        copy_mp_list(self._gates_prob, self.convert_to_gates_prob(num_exit_per_gate), self._gates_prob_lock)
        exit_gate = np.random.choice(self.gates_num, p=self.gates_prob)
        transmit, partition_layers, select_destination = self._transmit_action()
        action = ScheduleAction(single_gate=exit_gate,
                                transmit=transmit,
                                partition_layers=partition_layers,
                                transmit_destination=select_destination
                                )
        self.logger.log("INFO", f"Exit Gate: {exit_gate}")
        if self._record_gate_prob:
            self.logger.log("INFO", f"Gate prob: {self.gates_prob}")
        if self._record_confidence:
            self.logger.log("INFO", f"Gate confidence: {self.gates_confidence}")
        return action


class ProbBaseMode(BaseSingleExitScheduler):
    def __init__(self, model_profile, system_profile, maxsize, log_path=None, **kwargs):
        super(ProbBaseMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path, **kwargs)
        self._gates_prob = self.manager.list(kwargs.get('exit_prob', None))
        assert self._gates_prob is not None, "exit_prob must be provided"
        self._mode_name = 'ProbBaseMode'

    def get_info(self):
        info = [
            "Scheduler: SingleExitScheduler",
            "Mode: {}".format(self.mode),
            "Gates probability: {}".format(self.gates_prob),
        ]
        return "\n".join(info)

    @time_wrapper
    def schedule(self, **kwargs):
        exit_gate = np.random.choice(self.gates_num, p=self.gates_prob)
        transmit, partition_layers, select_destination = self._transmit_action()
        action = ScheduleAction(single_gate=exit_gate,
                                transmit=transmit,
                                partition_layers=partition_layers,
                                transmit_destination=select_destination
                                )
        self.logger.log("INFO", f"Exit Gate: {exit_gate}")
        return action


class ProbMultiMode(BaseSingleExitScheduler):
    def __init__(self, model_profile, system_profile, maxsize, log_path=None, **kwargs):
        super(ProbMultiMode, self).__init__(model_profile, system_profile, maxsize, log_path=log_path, **kwargs)
        self._gates_prob = self.manager.list(kwargs.get('exit_prob', None))
        assert self._gates_prob is not None, "exit_prob must be provided"
        self._mode_name = 'ProbMultiMode'
        self.sleep_time = kwargs.get('sleep_time', 0)

    def get_info(self):
        info = [
            "Scheduler: MultiExitScheduler",
            "Mode: {}".format(self.mode),
            "Gates probability: {}".format(self.gates_prob),
        ]
        return "\n".join(info)

    @time_wrapper
    def schedule(self, **kwargs):
        time.sleep(self.sleep_time)
        exit_gate = np.random.choice(self.gates_num, p=self.gates_prob)
        exit_threshold = np.ones(len(self.gates_prob))
        exit_threshold[exit_gate] = 0
        transmit, partition_layers, select_destination = self._transmit_action()
        action = ScheduleAction(multi_gates=exit_threshold,
                                transmit=transmit,
                                partition_layers=partition_layers,
                                transmit_destination=select_destination
                                )
        self.logger.log("INFO", f"Exit threshold: {exit_threshold}")
        return action


class SingleExitScheduler:
    MODES = {
        'random': RandomMode,
        'single': SingleMode,
        'specify': SpecifyMode,
        'reactive': ReactiveMode,
        'model_based': ModelBaseMode,
        'conf_model_based': ConfidenceModelBaseMode,
        'rate_based': RateBaseMode,
        'conf_rate_based': ConfidenceRateBaseMode,
        'prob_based': ProbBaseMode,
        'prob_multi': ProbMultiMode
    }

    def __init__(self, model_profile, system_profile, mode, maxsize: int = None, max_memory_size: int = None,
                 log_path=None, **kwargs):
        self.support_mode = list(self.MODES.keys())
        mode = mode.lower()
        self._validate_mode(mode)

        self._mode = self.MODES[mode](model_profile, system_profile, maxsize=maxsize, max_memory_size=max_memory_size,
                                      log_path=log_path, **kwargs)

    def __getattr__(self, item):
        return getattr(self._mode, item)

    def _validate_mode(self, mode):
        if mode not in self.MODES:
            supported_modes = ', '.join(self.MODES.keys())
            raise ValueError(f"mode must be in [{supported_modes}]")
