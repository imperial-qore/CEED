import argparse
import pickle
import json
import os.path

import numpy as np
from scipy.optimize import linprog, fminbound
from line_profiler import profile


class Result:
    def __init__(self, fun=None, x=None, success=False):
        if success:
            assert fun is not None and x is not None, "objective and solution must be provided"
        self.fun = fun
        self.x = x
        self.success = success


class ModelBaseOptimisation:
    def __init__(self, gates_process_time, max_queue_length, num_sample_p, loss_ratio, fast=False, **kwargs):
        self.max_queue_length = max_queue_length + 1
        self.num_sample_p = num_sample_p
        self.loss_ratio = loss_ratio
        self.lp_time = kwargs.get('lp_time', 0)
        self.fast = fast
        if self.fast:
            pt_min = min(gates_process_time)
            pt_max = max(gates_process_time)
            rho_start = kwargs.get('arrival_min', 0.1) * pt_min
            rho_end = kwargs.get('arrival_max', 100) * pt_max
            rho_sample = kwargs.get('samples', 1000)
            self.rho_s2_pair = self._init_fminbound(rho_start, rho_end, rho_sample)

    @profile
    def _init_fminbound(self, rho_start=0.1, rho_end=100, samples=1000):
        options = {'xatol': 1e-2, 'fatol': 1e-4, 'disp': 0}
        rho_s2_pair = dict()
        rhos = np.linspace(rho_start, rho_end, samples)
        sqrt_rhos = np.sqrt(rhos)
        for rho, sqrt_rho in zip(rhos, sqrt_rhos):
            s2_max = fminbound(self.objective, 0, 1024, args=[rho, sqrt_rho, self.max_queue_length, self.loss_ratio],
                               xtol=options['xatol'], disp=options['disp'])
            rho_s2_pair[rho] = s2_max
        return rho_s2_pair

    def _opt_rho(self, rho):
        keys = list(self.rho_s2_pair.keys())
        keys_array = np.array(keys)
        return keys[np.abs(keys_array - rho).argmin()]

    @staticmethod
    def loss(s, rho, sqrt_rho, max_queue_length):
        numerator = rho ** (
                (sqrt_rho * s ** 2 - sqrt_rho + 2 * max_queue_length) / (2 + sqrt_rho * s ** 2 - sqrt_rho)) * (
                            rho - 1)
        denominator = rho ** (
                2 * (1 + sqrt_rho * s ** 2 - sqrt_rho + max_queue_length) / (2 + sqrt_rho * s ** 2 - sqrt_rho)) - 1
        return numerator / denominator

    @staticmethod
    def solve_linear_programming(gates_process_time, gates_accuracy, pmean, s2):
        c = -1 * np.array(gates_accuracy)
        A_eq = np.array([[1] * len(gates_process_time), gates_process_time])
        b_eq = np.array([1, pmean])
        A_uq = np.square(np.array([gates_process_time]))
        b_uq = np.array([(1 + s2) * np.square(pmean)])
        bounds = [(0, 1)] * len(gates_process_time)
        # use scipy linprog to solve the linear programming problem
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_uq, b_ub=b_uq, bounds=bounds, method='highs')
        if res.success:
            result = Result(-res.fun, res.x, res.success)
        else:
            result = Result(success=False)
        return result

    @classmethod
    def objective(cls, x, rho, sqrt_rho, max_queue_length, loss_ratio):
        return np.square(cls.loss(x, rho, sqrt_rho, max_queue_length) - loss_ratio)

    def _fast_optimisation(self, gates_process_time, gates_accuracy, arrival_rate):
        gates_process_time = (np.array(gates_process_time) + self.lp_time).tolist()
        pt_min = min(gates_process_time)
        pt_max = max(gates_process_time)
        pmean = np.linspace(pt_min, pt_max, self.num_sample_p)
        acc_opt = -np.inf
        final_p_selected = [0] * len(gates_process_time)
        final_p_selected[0] = 1
        final_p_selected = np.array(final_p_selected)

        last_rho = np.nan

        for p in pmean:
            rho = arrival_rate * p
            rho = self._opt_rho(rho)
            if rho == last_rho:
                continue
            last_rho = rho
            sqrt_rho = np.sqrt(rho)
            s2_max = self.rho_s2_pair[rho]
            estimate_loss = self.loss(s2_max, rho, sqrt_rho, self.max_queue_length)
            if 0 <= estimate_loss <= self.loss_ratio:
                res = self.solve_linear_programming(gates_process_time, gates_accuracy, p, s2_max)
                if res.success:
                    p_selected = res.x
                    object_func = res.fun
                    if object_func > acc_opt:
                        final_p_selected = p_selected
                        acc_opt = object_func
        return final_p_selected

    def _full_optimisation(self, gates_process_time, gates_accuracy, arrival_rate):
        options = {'xatol': 1e-2, 'fatol': 1e-4, 'disp': 0}
        gates_process_time = (np.array(gates_process_time) + self.lp_time).tolist()
        pt_min = min(gates_process_time)
        pt_max = max(gates_process_time)
        pmean = np.linspace(pt_min, pt_max, self.num_sample_p)
        acc_opt = -np.inf
        final_p_selected = [0] * len(gates_process_time)
        final_p_selected[0] = 1
        final_p_selected = np.array(final_p_selected)

        for p in pmean:
            rho = arrival_rate * p
            sqrt_rho = np.sqrt(rho)

            s2_max = fminbound(self.objective, 0, 1024, args=[rho, sqrt_rho, self.max_queue_length, self.loss_ratio],
                               xtol=options['xatol'], disp=options['disp'])

            estimate_loss = self.loss(s2_max, rho, sqrt_rho, self.max_queue_length)
            if 0 <= estimate_loss <= self.loss_ratio:
                res = self.solve_linear_programming(gates_process_time, gates_accuracy, p, s2_max)
                if res.success:
                    p_selected = res.x
                    object_func = res.fun
                    if object_func > acc_opt:
                        final_p_selected = p_selected
                        acc_opt = object_func
        return final_p_selected

    def optimisation(self, gates_process_time, gates_accuracy, arrival_rate):
        """
        estimate the mean processing time and the squared coefficient of variation to achieve the desired loss ratio
        based on eq(11) in the paper
        :return:
        """
        if self.fast:
            return self._fast_optimisation(gates_process_time, gates_accuracy, arrival_rate)
        else:
            return self._full_optimisation(gates_process_time, gates_accuracy, arrival_rate)


def get_model_profile(model_path, machine_type):
    with open(f'{model_path}/{machine_type}/profile_summary.json', 'r') as f:
        system_profile = json.load(f)

    with open(f'{model_path}/config.json', 'r') as f:
        model_profile = json.load(f)
    exit_gates = model_profile['early_exits_gates']
    gates_process_time = list()
    gates_accuracy = list()
    for i, g in enumerate(exit_gates):
        process_time_info = system_profile.get(f"execution_time_gate_{i}_layer_{g}", None)
        if process_time_info is None:
            raise ValueError("model profile and system profile not match")
        gates_process_time.append(process_time_info['mean'])
        gates_accuracy.append(model_profile['accuracy'][f"gate_{i}"])
    return gates_process_time, gates_accuracy


def main(model_path, machine_type, arrival_rate, task_queue_size, num_sample_p, loss_ratio):
    """
    running linear programming to get the optimal gates probability to achieve the best accuracy within the desired
    loss ratio
    based on eq(12) in the paper
    :return:
    """

    gates_process_time, gates_accuracy = get_model_profile(model_path, machine_type)
    model = ModelBaseOptimisation(gates_process_time, task_queue_size, num_sample_p, loss_ratio)
    gates_prob = model.optimisation(gates_process_time,
                                    gates_accuracy,
                                    arrival_rate)
    return gates_prob


@profile
def test():
    model_path = '../../../../source/models/resnet50_bn_small_ee_cifar_10_end_join'
    machine_type = 'worker'
    arrival_rate = 1 / 0.2
    task_queue_size = 10
    num_sample_p = 100
    loss_ratio = 0.000650920587688302
    gates_process_time, gates_accuracy = get_model_profile(model_path, machine_type)
    model = ModelBaseOptimisation(gates_process_time, task_queue_size, num_sample_p, loss_ratio,
                                  fast=True, arrival_min=0.1, arrival_max=20, samples=2000)
    gates_prob = model.optimisation(gates_process_time,
                                    gates_accuracy,
                                    arrival_rate)
    print(gates_prob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model based scheduler')
    parser.add_argument('--model_path', type=str,
                        default='/home/chanyikchong/Work/Early-Exits-Deploy/Early_Exits_Deployment/source/models/resnet50_bn_small_ee_cifar_10_end_join')
    parser.add_argument('--machine_type', type=str, default='worker')
    parser.add_argument('--task_queue_size', type=int, default=10)
    parser.add_argument('--num_sample_p', type=int, default=100)
    parser.add_argument('--loss_ratio', type=float, default=0.000650920587688302)
    parser.add_argument("--int", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true",
                        help="run 'kernprof -l -v generate_model_base_lookup_table.py --test' to test the function")
    hp = parser.parse_args()

    if hp.test:
        test()
        exit(0)

    lookup_table = dict()
    arrival_rates = np.load(os.path.join(hp.model_path, 'lookup_arrival.npy'))
    for arrival_rate in arrival_rates:
        if hp.int:
            poisson_arrival_rate = arrival_rate
        else:
            poisson_arrival_rate = 1 / arrival_rate
        gates_prob = main(hp.model_path, hp.machine_type, poisson_arrival_rate, hp.task_queue_size, hp.num_sample_p,
                          hp.loss_ratio)
        lookup_table[arrival_rate] = gates_prob.tolist()

    save_folder = os.path.join(hp.model_path, hp.machine_type)
    save_folder = 'backup'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, 'model_based_lookup_table.pkl'), 'wb') as f:
        pickle.dump(lookup_table, f)
