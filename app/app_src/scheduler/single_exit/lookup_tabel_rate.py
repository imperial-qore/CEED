import argparse
import json
import os.path
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import sys
sys.path.append("<project root path>")
from app.app_src.scheduler.single_exit.utils.branch_n_bound import KnapSackBranchNBound


class RateLookupTable:
    def __init__(self, model_path, machine_type, max_size, arrival_max=8, arrival_min=1, n_sample_arrival=1000):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            self.model_profile = json.load(f)

        if machine_type == 'worker':
            sys_path = os.path.join(model_path, 'workers', 'worker0', "profile_summary.json")
        elif machine_type == 'edge':
            sys_path = os.path.join(model_path, 'edges', 'edge0', "profile_summary.json")
        elif machine_type == 'cloud':
            sys_path = os.path.join(model_path, 'clouds', 'minipc', "profile_summary.json")

        with open(sys_path, "r") as f:
            self.system_profile = json.load(f)

        self.exit_gates = self.model_profile['early_exits_gates']
        self.gates_process_time = list()
        self.gates_accuracy = list()
        for i, g in enumerate(self.exit_gates):
            process_time_info = self.system_profile.get(f"execution_time_gate_{i}_layer_{g}", None)
            if process_time_info is None:
                raise ValueError("model profile and system profile not match")
            self.gates_process_time.append(process_time_info['mean'])
            self.gates_accuracy.append(self.model_profile['accuracy'][f"gate_{i}"])
        self.max_size = max_size
        self.arrival_max = arrival_max
        self.arrival_min = arrival_min
        self.n_sample_arrival = n_sample_arrival
        self.arrivals = np.linspace(arrival_min, arrival_max, n_sample_arrival)

    def time_budget(self, arrival_rate, current_queue):
        budge = arrival_rate * (self.max_size - current_queue)
        return budge

    def generate_table(self):
        table = defaultdict(dict)
        max_size = self.max_size
        gap = 1
        if max_size > 10000:
            max_size = max_size//10
            gap = 10
        if max_size > 10000:
            max_size = max_size//10
            gap = 100

        for arrival in tqdm(self.arrivals):
            arrival = 1 / arrival
            can_skip = False
            for n_job in tqdm(range(1, max_size)):
                if can_skip:
                    num_exit_per_gate = np.zeros(len(self.gates_process_time), dtype=int)
                    num_exit_per_gate[0] = n_job
                    table[arrival][n_job] = num_exit_per_gate.tolist()
                    continue

                n_job = n_job * gap
                budget_constrain = self.time_budget(arrival, n_job) - n_job * self.gates_process_time[0]  # W
                items = [(acc, pt) for acc, pt in zip(self.gates_accuracy[1:], self.gates_process_time[1:])]
                _, num_exit_per_gate = KnapSackBranchNBound.fit(budget_constrain, items, n_job)
                if num_exit_per_gate is None:
                    num_exit_per_gate = np.zeros(len(self.gates_process_time), dtype=int)
                    num_exit_per_gate[0] = n_job
                else:
                    num_exit_per_gate = [n_job - int(sum(num_exit_per_gate))] + num_exit_per_gate
                    num_exit_per_gate = np.array(num_exit_per_gate, dtype=int)
                    if num_exit_per_gate[0] == n_job:
                        can_skip = True

                table[arrival][n_job] = num_exit_per_gate
        return table


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', type=str,
                      default='../../../../source/models/resnet50_bn_small_ee_cifar_10_end_join')
    args.add_argument('--machine_type', type=str, default='worker')
    args.add_argument('--max_size', type=int, default=10)
    args.add_argument("--arrival_max", type=float, default=8)
    args.add_argument("--arrival_min", type=float, default=1)
    args.add_argument("--n_sample_arrival", type=int, default=20)
    args = args.parse_args()

    lookup_table = RateLookupTable(args.model_path, args.machine_type, args.max_size, args.arrival_max,
                                   args.arrival_min, args.n_sample_arrival)
    table = lookup_table.generate_table()

    if args.machine_type == 'worker':
        save_path = os.path.join(args.model_path, 'worker_lookup_table.pkl')
    elif args.machine_type == 'edge':
        save_path = os.path.join(args.model_path, 'edge_lookup_table.pkl')
    elif args.machine_type == 'cloud':
        save_path = os.path.join(args.model_path, 'cloud_lookup_table.pkl')

    with open(save_path, "wb") as f:
        pickle.dump(table, f)
