from typing import Union, Dict, List, Optional
from multiprocessing import Manager

import numpy as np
from smt.sampling_methods import LHS
from tqdm import tqdm
from dataclasses import dataclass

from ...utils import process_time_profile, get_maximum_process_time, get_machine_type

from ..scheduler import Scheduler, ScheduleAction, time_wrapper


@dataclass
class AdaEEUpdate:
    queue_idx: int = None
    arm_idx: int = None
    early_exit: bool = False
    early_exit_layer: Optional[int] = None
    final_confidence: Optional[float] = None
    early_exit_confidence: float = None
    early_process_time: float = None
    final_process_time: Optional[float] = None


class UCB:
    """Upper Confidence Bound (UCB) for dynamic decision making."""

    def __init__(self, thresholds, q_idx=0, manager=None):
        self.q_idx = q_idx
        self.n_arms = len(thresholds)
        self.select_boolen = manager.list([0] * self.n_arms)  # Store if the arm has been selected
        self.counts = manager.list([0] * self.n_arms)  # Number of times each arm is selected
        self.values = manager.list([0.0] * self.n_arms)  # Estimated reward per arm
        self.thresholds = thresholds  # Store thresholds for this queue size
        self.select_ubc_count = manager.Value('i', 0)

    def select_arm(self):
        """ Selects an arm using UCB formula. """
        total_counts = sum(self.counts)
        # make sure every arm has run once
        zeros = [i for i, b in enumerate(self.select_boolen) if b == 0]
        if zeros:
            select_arm = zeros[0]
        else:
            ucb_values = [v + np.sqrt(2 * np.log(total_counts + 1e-6) / (c + 1e-5))
                          for v, c in zip(self.values, self.counts)]
            select_arm = int(np.argmax(ucb_values))
            # with self.select_ubc_count.get_lock():
            self.select_ubc_count.value += 1
        self.select_boolen[select_arm] = 1
        return select_arm

    def update(self, arm, reward):
        """ Updates UCB estimates with a new reward. """
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class AdaEEBuffer(Scheduler):
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
        super(AdaEEBuffer, self).__init__(model_profile, system_profile, maxsize, max_memory_size,
                                          arrival_estimate_window,
                                          log_path, **kwargs)
        self._name = 'AdaEEBuffer'
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
            self.action_set = {q: {i: a for i, a in enumerate(action_set)} for q in range(self.maxsize)}
        elif isinstance(action_set, dict):
            action_set = {q: self._generate_valid_action(action_set[q]) for q in range(self.maxsize)}
            self.action_set = action_set
        else:
            self.action_set = action_set

        self.ucb_instances = {q: UCB(self.action_set[q], q_idx=q, manager=manager) for q in range(self.maxsize)}
        self.skip_exit_threshold = kwargs.get("skip_exit_threshold", 1)
        self.maximum_process_time = get_maximum_process_time(system_profile)

        self.process_time_config = process_time_profile(self.system_profile, self._gates, self.active_gates,
                                                        normalize=True)
        self.early_exit_process_time = self.process_time_config[self.active_gates[0]]['mean']
        self.final_exit_process_time = self.process_time_config[-1]['mean']

    @property
    def scheduler_type(self):
        return "adaee_buffer"

    def get_info(self):
        return "AdaEEBuffer scheduler"

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
        ubc = self.ucb_instances[self.q_size]
        arm = ubc.select_arm()
        action = ubc.thresholds[arm]
        transmit, partition_layers, select_destination = self._transmit_action()
        schedule_action = ScheduleAction(multi_gates=action,
                                         skip_exit_threshold=self.skip_exit_threshold,
                                         transmit=transmit,
                                         partition_layers=partition_layers,
                                         transmit_destination=select_destination,
                                         queue_idx=self.q_size,
                                         arm_idx=arm)
        self.logger.log("INFO", f"Exit Threshold: {action}")
        return schedule_action

    def update(self, adaee_buffer_update: AdaEEUpdate, **kwargs):
        # update the rewards
        queue_idx = adaee_buffer_update.queue_idx
        arm_idx = adaee_buffer_update.arm_idx
        if adaee_buffer_update.early_exit:
            reward = 0  # processing time of early exit
        else:
            extra_processing_time = adaee_buffer_update.final_process_time - adaee_buffer_update.early_process_time  # processing time of final exit

            early_exit_confidence = adaee_buffer_update.early_exit_confidence
            final_confidence = adaee_buffer_update.final_confidence

            reward = np.max((final_confidence - early_exit_confidence), 0) - (1 / (10 * self.maxsize)) * queue_idx - (
                    1 / 10000) * extra_processing_time

        previous_value = self.ucb_instances[queue_idx].values[arm_idx]
        self.ucb_instances[queue_idx].update(arm_idx, reward)
        self.logger.log('info', f"Reward Update: {queue_idx}_{arm_idx}: {previous_value} -> {self.ucb_instances[queue_idx].values[arm_idx]}")
