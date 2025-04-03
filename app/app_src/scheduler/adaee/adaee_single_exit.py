from typing import Union, Dict, List, Optional
from multiprocessing import Manager

import numpy as np
from smt.sampling_methods import LHS
from tqdm import tqdm

from .adaee_buffer import UCB, AdaEEUpdate
from ...utils import process_time_profile, get_maximum_process_time, get_machine_type
from ..scheduler import Scheduler, ScheduleAction, time_wrapper


class AdaEEOrigin(Scheduler):
    def __init__(
            self,
            model_profile: Union[str, Dict],
            system_profile: Union[str, Dict],
            maxsize: int = None,
            arrival_estimate_window: int = None,
            log_path: str = None,
            n_action_options: int = 3,
            action_set: List = None,
            active_gates: List[int] = None,
            **kwargs
    ):
        max_memory_size = None  # only support queue length
        super(AdaEEOrigin, self).__init__(model_profile, system_profile, maxsize, max_memory_size,
                                          arrival_estimate_window, log_path, **kwargs)
        self._name = 'AdaEEOrigin'
        manager = Manager()

        # the number of option for each gate. It must be greater or equal to 2
        # Because the gate is full open or close or open with probability
        assert n_action_options >= 2
        self.n_action_options = len(action_set) if action_set else n_action_options
        self.active_gates = active_gates if active_gates else [i for i in range(len(self._gates) - 1)]
        # initialize actions and rewards
        if action_set is None:
            actions = self._generate_actions()
            self.action_set = {i: a for i, a in enumerate(actions)}
        elif isinstance(action_set, list):
            action_set = self._generate_valid_action(action_set)
            self.action_set = {i: a for i, a in enumerate(action_set)}
        elif isinstance(action_set, dict):
            action_set = self._generate_valid_action(action_set)
            self.action_set = action_set
        else:
            self.action_set = action_set

        self.ucb_instances = UCB(self.action_set, q_idx=0, manager=manager)
        self.skip_exit_threshold = kwargs.get("skip_exit_threshold", 1)
        self.maximum_process_time = get_maximum_process_time(system_profile)

        self.process_time_config = process_time_profile(self.system_profile, self._gates, self.active_gates,
                                                        normalize=True)

    @property
    def scheduler_type(self):
        return "adaee_origin"

    def get_info(self):
        return "AdaEEOrigin Scheduler"

    def _generate_valid_action(self, action_set):
        action_set = np.array(action_set) if not isinstance(action_set, np.ndarray) else action_set
        if action_set.ndim == 1:
            action_set = action_set.reshape(-1, 1)
        actions = np.ones((self.n_action_options, len(self._gates)))
        actions[:, self.active_gates] = action_set
        actions[:, -1] = 0
        return actions

    def _generate_actions(self) -> List[List[float]]:
        # each gate except the final gate has n options
        low_bound = 0.01
        upper_bound = 1.01
        gate_with_options = len(self.active_gates)
        # values = np.linspace(low_bound, upper_bound, self.n_action_options)
        # actions = np.array(list(itertools.product(values, repeat=gate_with_options)))

        # Use LHS to generate actions
        action_sampler = LHS(xlimits=np.array([[low_bound, upper_bound]] * gate_with_options), criterion='ese',
                             random_state=42)
        actions = list()
        count = 0
        while len(actions) < self.n_action_options and count < 1000:
            actions_sample = action_sampler(self.n_action_options)
            if len(self.active_gates) < len(self._gates) - 1:
                actions_sample = self._complete_actions(actions_sample)
            actions_sample = [np.concatenate([action, np.array([0])]) for action in
                              tqdm(actions_sample, desc="validate actions") if
                              self._is_valid(action)]
            actions.extend(actions_sample)
            count += 1
        return actions

    def _complete_actions(self, actions):
        """
        complete the generated actions if activated gates are provided
        :param actions:
        :return:
        """
        num_actions, action_dim = actions.shape
        action_matrix_size = len(self._gates) - 1
        completed_actions = np.ones((num_actions, action_matrix_size))
        for i, col_index in enumerate(tqdm(self.active_gates, desc="completing actions")):
            completed_actions[:, col_index] = actions[:, i]
        return completed_actions

    @staticmethod
    def _is_valid(combination):
        for i in range(len(combination) - 1):
            # if a gate is open then only keep the action which the remaining gates are all opened
            if combination[i] <= 0 and combination[i] != combination[i + 1]:
                return False
        return True

    @time_wrapper
    def schedule(self, **kwargs) -> ScheduleAction:
        ubc = self.ucb_instances
        arm = ubc.select_arm()
        action = ubc.thresholds[arm]
        transmit, partition_layers, select_destination = self._transmit_action()
        schedule_action = ScheduleAction(multi_gates=action,
                                         skip_exit_threshold=self.skip_exit_threshold,
                                         transmit=transmit,
                                         partition_layers=partition_layers,
                                         transmit_destination=select_destination,
                                         queue_idx=0,
                                         arm_idx=arm)
        self.logger.log("INFO", f"Exit Threshold: {action}")
        return schedule_action

    def update(self, adaee_buffer_update: AdaEEUpdate, **kwargs):
        # update the rewards
        arm_idx = adaee_buffer_update.arm_idx
        if adaee_buffer_update.early_exit:
            reward = 0  # processing time of early exit
        else:
            extra_processing_time = adaee_buffer_update.final_process_time - adaee_buffer_update.early_process_time  # processing time of final exit

            early_exit_confidence = adaee_buffer_update.early_exit_confidence
            final_confidence = adaee_buffer_update.final_confidence

            reward = np.max((final_confidence - early_exit_confidence), 0) - (
                    1 / 10000) * extra_processing_time

        previous_value = self.ucb_instances.values[arm_idx]
        self.ucb_instances.update(arm_idx, reward)
        self.logger.log('info', f"Reward Update: {arm_idx}: {previous_value} -> {self.ucb_instances.values[arm_idx]}")
