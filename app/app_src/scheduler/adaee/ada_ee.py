import pickle
from typing import Union, Dict, List
import itertools
import socket

import numpy as np
from tqdm import tqdm
from smt.sampling_methods import LHS

from ...utils import process_time_profile, get_maximum_process_time, get_machine_type
from ..scheduler import Scheduler, ScheduleAction, time_wrapper


class AdaEE(Scheduler):
    def __init__(self,
                 model_profile: Union[str, Dict],
                 system_profile: Union[str, Dict],
                 maxsize: int = None,
                 max_memory_size: int = None,
                 arrival_estimate_window: int = None,
                 log_path: str = None,
                 learning_rate: float = 0.5,
                 reward_mode: str = "avg",
                 n_action_options: int = 3,
                 action_set: List = None,
                 rewards: list = None,
                 n_memory: list = None,
                 scheduler_config: Union[str, Dict] = None,
                 active_gates: List[int] = None,
                 **kwargs):
        """
        AdaEE: Adaptive Early Exit Scheduler
        Standard Scheduler parameters:
        :param model_profile:
        :param system_profile:
        :param maxsize:
        :param log_path:

        Param Option 1:
        initialize the scheduler with the given parameters for initialization
        :param learning_rate:
        :param reward_mode:
        :param n_action_options:

        Param Option 2:
        initialize the scheduler with the given trained parameters
        :param learning_rate:
        :param action_set:
        :param rewards:
        :param n_memory:

        Param Option 3:
        initialize the scheduler with the given scheduler config file
        :param scheduler_config:

        :param kwargs:
        """
        super(AdaEE, self).__init__(model_profile, system_profile, maxsize, max_memory_size, arrival_estimate_window,
                                    log_path, **kwargs)
        self._name = 'AdaEE'
        self.initialized = True

        # load the scheduler config
        if scheduler_config:
            print("loading scheduler config")
            if isinstance(scheduler_config, str):
                hostname = socket.gethostname()
                machine_type = get_machine_type(hostname)
                scheduler_config = scheduler_config.format(machine_type=machine_type, machine_name=hostname)
                with open(scheduler_config, "rb") as f:
                    scheduler_config = pickle.load(f)
            learning_rate = scheduler_config["learning_rate"]
            reward_mode = scheduler_config["reward_mode"]
            n_action_options = scheduler_config["n_action_options"]
            action_set = scheduler_config["action_set"]
            rewards = scheduler_config["rewards"]
            n_memory = scheduler_config["n_memory"]

        self.learning_rate = learning_rate
        assert reward_mode in ["avg", "max"]
        self.reward_mode = reward_mode
        self.maximum_process_time = get_maximum_process_time(system_profile)

        # the number of option for each gate. It must be greater or equal to 2
        # Because the gate is full open or close or open with probability
        assert n_action_options >= 2
        self.n_action_options = n_action_options
        self.active_gates = active_gates if active_gates else [i for i in range(len(self._gates) - 1)]

        # initialize actions and rewards
        if action_set is None:
            actions = self._generate_actions()
            self.action_set = {i: a for i, a in enumerate(actions)}
        elif isinstance(action_set, list):
            self.action_set = {i: a for i, a in enumerate(action_set)}
        else:
            self.action_set = action_set

        # initialize rewards and n_memory which is Q_t and N_t in the paper
        if rewards is None:
            rewards = [0] * len(self.action_set)
            self.initialized = False
        self.rewards = np.array(rewards, dtype=np.float32)

        self.total_t = 0
        if n_memory is None:
            n_memory = [0] * len(self.action_set)
        self.n_memory = np.array(n_memory)
        self.total_t = np.sum(self.n_memory)
        self.skip_exit_threshold = kwargs.get("skip_exit_threshold", np.inf)

    @property
    def scheduler_type(self):
        return "adaee"

    def get_info(self):
        return "AdaEE scheduler"

    @time_wrapper
    def schedule(self, **kwargs) -> ScheduleAction:
        # get the UCB of each action
        actions_reward = self.rewards + self.learning_rate * np.sqrt(np.log(self.total_t) / self.n_memory)
        action_idx = np.argmax(actions_reward)
        action = self.action_set[action_idx].tolist()
        transmit, partition_layers, select_destination = self._transmit_action()
        schedule_action = ScheduleAction(multi_gates=action,
                                         action_idx=action_idx,
                                         skip_exit_threshold=self.skip_exit_threshold,
                                         transmit=transmit,
                                         partition_layers=partition_layers,
                                         transmit_destination=select_destination)
        self.logger.log("INFO", f"Exit Threshold: {action}")
        return schedule_action

    def update(self, scheduler_action, confidences=None, update_n=True, **kwargs):
        # update the rewards
        if confidences is not None:
            reward = self._calculate_reward(scheduler_action['multi_gates'], confidences)
            self.rewards[scheduler_action['action_idx']] = (reward + self.n_memory[scheduler_action['action_idx']] *
                                                            self.rewards[
                                                                scheduler_action['action_idx']]) / (
                                                                   self.n_memory[scheduler_action['action_idx']] + 1)
            self.n_memory[scheduler_action['action_idx']] += 1
            if update_n:
                # update the total time
                self.total_t += 1

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
                actions = self._complete_actions(actions_sample)
            actions_sample = [np.concatenate([action, np.array([0])]) for action in
                              tqdm(actions_sample, desc="validate actions") if
                              self._is_valid(action)]
            actions.extend(actions_sample)
            count += 1
        return actions

    @staticmethod
    def _is_valid(combination):
        for i in range(len(combination) - 1):
            # if a gate is open then only keep the action which the remaining gates are all opened
            if combination[i] <= 0 and combination[i] != combination[i + 1]:
                return False
        return True

    def _calculate_reward(self, threshold, confidences):
        confidences = np.array(confidences).flatten()
        activate_gates_idx = np.where(np.array(threshold) < 1.0)[0].tolist()
        time_profile = process_time_profile(self.system_profile, self._gates,
                                            activated_gates_idx=activate_gates_idx, normalize=self.maximum_process_time)
        time_profile.pop(-1)  # exclude the transmit time
        time_profile = dict(sorted(time_profile.items()))

        gate_indices = list(time_profile.keys())
        last_exit_index = len(confidences) - 1
        last_exit_id = gate_indices[last_exit_index]

        confidence_gain = np.maximum(confidences[last_exit_index] - np.array(confidences[:last_exit_index]), 0)
        time_gain = [time_profile[last_exit_id]['mean'] - time_profile[i]['mean'] for i in gate_indices if
                     i < last_exit_id]
        rewards = confidence_gain - np.array(time_gain)

        # The original paper only consider one exit gate in their model. In order to address it to multiple exit gates,
        # we use different method to summarize the reward.
        if len(rewards) == 0:
            return 0

        if self.reward_mode == "avg":
            # take the average of rewards as the final reward
            return np.mean(rewards)
        elif self.reward_mode == 'max':
            # take the maximum of rewards as the final reward
            return np.max(rewards)
        else:
            raise ValueError(f"invalid reward mode {self.reward_mode}")

    def initialized_scheduler(self, confidences, verbose=False):
        print("Start initializing scheduler")

        for action_idx, action in tqdm(self.action_set.items(), disable=not verbose):
            scheduler_action = {
                "multi_gates": action,
                "action_idx": action_idx,
            }
            # predict = model.inference(image, skip_exit_threshold=1.0, **scheduler_action)
            activate_gates_idx = np.where(np.array(action) < 1.0)[0].tolist()
            select_confidences = confidences[activate_gates_idx]
            activated_action = action[activate_gates_idx]
            exit_gate_idx = np.where(select_confidences > activated_action)[0][0]
            select_confidences = confidences[:exit_gate_idx + 1]
            self.update(scheduler_action, select_confidences, update_n=False)
        self.initialized = True
        print("Finish initializing scheduler")
