import socket
import pickle

import numpy as np
from ..scheduler import Scheduler, ScheduleAction, time_wrapper
from ...utils import get_machine_type


class BOOfflineScheduler(Scheduler):
    def __init__(self, model_profile, system_profile, maxsize: int = None, max_memory_size: int = None,
                 arrival_estimate_window: int = None, log_path: str = None, **kwargs):
        """
        Bayesian Optimisation Offline scheduler
        # standard scheduler parameters
        :param model_profile:
        :param system_profile:
        :param maxsize:
        :param log_path:
        :param kwargs:
        # require parameters
        :key initial_arrival_rate: initial arrival rate. For the first few tasks, the scheduler will use this rate
        :key lookup_table: the path of the lookup table. Should be a pickle file which contains a dictionary
        :key skip_exit_threshold: the threshold for skipping early exits
        """
        super(BOOfflineScheduler, self).__init__(model_profile, system_profile, maxsize, max_memory_size, arrival_estimate_window,
                                                 log_path, **kwargs)
        self.initial_arrival_rate = kwargs.get('initial_arrival_rate', 0.5)
        self.skip_exit_threshold = kwargs.get('skip_exit_threshold', np.inf)
        self.gates_layer = self.model_profile.get('early_exits_gates', list())

        lookup_table = kwargs.get('lookup_table', None)
        if lookup_table is None:
            raise ValueError("Lookup table is not provided")

        hostname = socket.gethostname()
        machine_type = get_machine_type(hostname)
        lookup_table = lookup_table.format(machine_type=machine_type, machine_name=hostname)
        with open(lookup_table, "rb") as f:
            self.arrival_lookup_table = pickle.load(f)

        self.arrival_rates = np.array(list(self.arrival_lookup_table.keys()))

    @property
    def scheduler_type(self):
        return "bo_offline"

    def get_info(self):
        return "Bayesian Optimisation Offline scheduler"

    @time_wrapper
    def schedule(self, **kwargs) -> ScheduleAction:
        if len(self._fcr.arrival_window) < min(self._fcr.max_size, 10) // 2:
            arrival_rate = self.initial_arrival_rate
        else:
            arrival_rate = self._fcr.estimate_arrival()
        mini_dist_arrival_pos = np.argmin(np.abs(self.arrival_rates - arrival_rate))
        select_arrival = self.arrival_rates[mini_dist_arrival_pos].astype(float)
        threshold = self.arrival_lookup_table[select_arrival]
        if len(threshold) == len(self.gates_layer) - 1:
            threshold.append(0)
        transmit, partition_layers, select_destination = self._transmit_action()
        action = ScheduleAction(multi_gates=threshold,
                                skip_exit_threshold=self.skip_exit_threshold,
                                transmit=transmit,
                                partition_layers=partition_layers,
                                transmit_destination=select_destination
                                )
        self.logger.log("INFO", f"Exit Threshold: {threshold}")
        return action
