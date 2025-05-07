import json
import pickle
import os
import sys
import time
from functools import partial
import shutil

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from bayes_opt import UtilityFunction

from .bayesian_optimization import CustomBayesianOptimization
from ..scheduler import ScheduleAction
from ...utils.utils import process_time_profile, get_maximum_process_time

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../../")))

from neural_network.nn.model import load_model
from neural_network.data.datasets import load_dataset
from neural_network.data.transform import set_transform


# cache_dir = '../cache'
# os.makedirs(cache_dir, exist_ok=True)


class BOThresholdTunner:
    def __init__(self, model_path: str, system_profile: str, cache_dir: str, seed: int = None, objective: str = None,
                 **kwargs):
        assert objective in ['loss_ratio',
                             'process_time',
                             'loss_ratio_regularize',
                             'loss_ratio_regularize_two_side'], f"Objective {objective} is not supported"
        self.model_path = model_path
        self.system_profile = system_profile
        self.cache_dir = cache_dir
        self.activated_gates_idx = kwargs.get('gates', None)

        self.mini_threshold = kwargs.get('mini_threshold', 0.01)
        self.max_threshold = kwargs.get('max_threshold', 1.0)

        self.objective = objective
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.alpha = kwargs.get('alpha', 0.5)
        self.preset_loss_ratio = kwargs.get('loss_ratio', 0)

        # this arrival rate is the lambda of poisson distribution
        self.arrival_rate = kwargs.get('arrival_rate', 5)
        self.max_queue_length = kwargs.get('max_queue_length', 10)
        self.penalty = kwargs.get('penalty', 100)

        self.model = None
        self.load_model()
        self.random_generator = np.random.RandomState(seed)
        self.max_process_time = get_maximum_process_time(self.system_profile)
        if self.activated_gates_idx is None:
            self.activated_gates_idx = list(range(len(self.model.gates_layer) - 1))

    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path
        self.model = load_model(self.model_path)
        self.model.eval()

    def load_data(self,
                  dataset_cache_dir,
                  dataset_name,
                  batch_size,
                  split='test',
                  valid_mode='valid',
                  random_seed=0,
                  ratio=0.5,
                  ):
        transform = self.model.transform
        train_dataset = load_dataset(dataset_name, split=split, streaming=False, cache_dir=dataset_cache_dir,
                                     valid_mode=valid_mode, seed=random_seed, valid_ratio=ratio)
        train_dataset.set_transform(partial(set_transform, transform=transform))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    def reload_cache_folder(self):
        if os.listdir(self.cache_dir):
            while True:
                is_repredict = input("Cache folder is not empty, do you want to re-predict? (y/n)")
                if is_repredict.lower() in ['y', 'n']:
                    break
            if is_repredict == 'n':
                return False
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        return True

    def prepare_prediction(self, dataloader):
        for i, batch in enumerate(tqdm(dataloader)):
            try:
                x, y = batch['img'], batch['label']
            except KeyError:
                x, y = batch['image'], batch['label']
            with torch.no_grad():
                y_out = self.model(x)
            if torch.any(y_out > 1) or torch.any(y_out < 0):
                y_out = torch.softmax(y_out, dim=1)
            confidence = self.model.confidence_func(y_out)
            save = {'likelihood': confidence, 'prediction': y_out.argmax(dim=1), 'label': y}
            torch.save(save, os.path.join(self.cache_dir, f"{i}.pt"))

    def evaluate_threshold(self, dataloader, threshold, save=False):
        """
        Evaluate the performance of a given threshold
        :param dataloader:
        :param threshold:
        :param save:
        :return:
        """
        if isinstance(threshold, tuple):
            threshold = list(threshold)

        assert len(threshold) >= len(self.model.gates_layer) - 1
        if len(threshold) == len(self.model.gates_layer) - 1:
            threshold.append(0)

        activate_gates_idx = torch.where(torch.tensor(threshold) < 1.0)[0].numpy().tolist()

        process_time_dict = process_time_profile(self.system_profile, gates_layer_idx=self.model.gates_layer,
                                                 activated_gates_idx=activate_gates_idx, normalize=False)

        schedule_action = ScheduleAction(multi_gates=threshold)
        process_time = 0
        process_time_list = list()
        acc = 0
        total = 0
        for batch in tqdm(dataloader):
            xs = batch['img']
            ys = batch['label']
            for x, y in zip(xs, ys):
                y_hat = self.model.inference(x, **schedule_action)
                exit_gate = self.model.gates_layer.index(y_hat.exit_layer)
                p_t = self.random_generator.normal(loc=process_time_dict[exit_gate]['mean'],
                                                   scale=process_time_dict[exit_gate]['var'])
                process_time += p_t
                process_time_list.append(p_t)
                if y == y_hat.output:
                    acc += 1
            total += ys.shape[0]
        acc /= total
        mean_process_time = process_time / total
        if save:
            with open(os.path.join(self.model_path, 'config.json'), 'r+', encoding='utf8') as f:
                model_config = json.load(f)
                model_config['best_possible_accuracy'] = acc
                f.seek(0)
                json.dump(model_config, f, indent=4)
                f.truncate()
        # plt.hist(process_time_list, bins=100)
        # plt.show()
        return acc, mean_process_time

    def threshold_tuning(self,
                         util_kind='ucb',
                         init_points=5,
                         n_iters=100,
                         early_stop_count=None,
                         seed=None,
                         save=False,
                         process_verbose=2):
        """
        Run Bayesian Optimization to find the optimal threshold
        :param util_kind:
        :param init_points:
        :param n_iters:
        :param early_stop_count:
        :param seed:
        :param save:
        :param process_verbose:
        :return:
        """
        early_stop_count = early_stop_count if early_stop_count else np.inf

        # select objective function
        if self.objective == 'loss_ratio':
            objective_function = self._objective_with_loss_ratio()
        elif self.objective == 'process_time':
            objective_function = self._objective_with_process_time()
        elif self.objective == 'loss_ratio_regularize':
            objective_function = self._objective_loss_ratio_with_regularize()
        elif self.objective == 'loss_ratio_regularize_two_side':
            objective_function = self._objective_loss_ratio_with_regularize_two_side()
        else:
            raise ValueError(f"Objective {self.objective} is not supported")

        n_gates = len(self.model.gates_layer)
        if self.activated_gates_idx:
            activated_gates_idx = self.activated_gates_idx
        else:
            activated_gates_idx = list(range(n_gates - 1))

        # set search bounds
        bounds = {f'x_{i}': (self.mini_threshold, self.max_threshold) for i in
                  range(n_gates - 1) if i in activated_gates_idx}
        optimizer = CustomBayesianOptimization(
            f=objective_function,
            pbounds=bounds,
            random_state=seed,
            allow_duplicate_points=True,
            verbose=process_verbose
        )
        acquisition_function = UtilityFunction(kind=util_kind, kappa=2.576, xi=0, kappa_decay=1, kappa_decay_delay=0)

        # start optimization
        start_time = time.perf_counter()
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iters,
            early_stop_count=early_stop_count,
            acquisition_function=acquisition_function
        )
        end_time = time.perf_counter()

        print(f"Optimal: {optimizer.max}, Time: {end_time - start_time}, Improvement: {optimizer.improvement}")
        if save:
            file_path = os.path.join(self.model_path, 'thresholds.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    thresholds = pickle.load(f)
            else:
                thresholds = dict()
            key = (self.arrival_rate, self.penalty)
            thresholds[key] = list(optimizer.max['params'].values())
            with open(file_path, 'wb') as f:
                pickle.dump(thresholds, f)
        return optimizer.max

    def _black_box_function(self, **kwargs):
        threshold = list(kwargs.values())

        threshold = [threshold[self.activated_gates_idx.index(i)] if i in self.activated_gates_idx else self.max_threshold for i in
                     range(len(self.model.gates_layer) - 1)]

        # if len(threshold) < len(self.model.gates_layer) - 1:
        #     threshold.extend([0] * (len(self.model.gates_layer) - 1 - len(threshold)))
        # # the threshold of the last gate is always 0
        threshold.append(0)
        threshold = torch.tensor(threshold)

        activate_gates_idx = torch.where(threshold < 1.0)[0].numpy().tolist()

        process_time_dict = process_time_profile(self.system_profile, gates_layer_idx=self.model.gates_layer,
                                                 activated_gates_idx=activate_gates_idx, normalize=False)

        data_files = os.listdir(self.cache_dir)
        total = 0
        acc = 0
        process_time_list = list()

        for p in data_files:
            save = torch.load(os.path.join(self.cache_dir, p))
            y_hat = save['likelihood']
            prediction = save['prediction']
            label = save['label']

            # get the first gate which likelihood is greater than the corresponding threshold
            indices = (y_hat >= threshold).max(dim=1)[1]
            # get the prediction of the gate with the indices
            prediction = prediction[torch.arange(prediction.shape[0]), indices]
            # get the process time of the gate with the indices
            process_time = [
                self.random_generator.normal(loc=process_time_dict[i.item()]['mean'],
                                             scale=process_time_dict[i.item()]['var'])
                for i in indices]

            process_time_list.extend(process_time)
            acc += torch.eq(prediction, label).sum().item()
            total += prediction.shape[0]

        acc /= total
        mean_process_time = np.mean(process_time_list)
        var_process_time = np.var(process_time_list)
        return acc, mean_process_time, var_process_time

    def _objective_with_process_time(self):
        def process_time_objective(**kwargs):
            acc, mean_process_time, _ = self._black_box_function(**kwargs)

            mean_process_time = mean_process_time / self.max_process_time

            # maximize accuracy and minimize process time with alpha as the parameter
            # to balance execution time and accuracy
            objective_function = self.alpha * acc - (1 - self.alpha) * mean_process_time
            return objective_function

        return process_time_objective

    def _estimate_loss_ratio(self, mean_process_time, var_process_time):
        s2 = var_process_time / mean_process_time ** 2
        rho = self.arrival_rate * mean_process_time
        sqrt_rho = np.sqrt(rho)

        a = sqrt_rho * s2 - sqrt_rho + 2 * self.max_queue_length
        b = 2 + sqrt_rho * s2 - sqrt_rho
        c = 1 + sqrt_rho * s2 - sqrt_rho + self.max_queue_length
        d = 2 + sqrt_rho * s2 - sqrt_rho
        d_hat = (rho ** np.divide(a, b) * (rho - 1)) / (rho ** (2 * np.divide(c, d)) - 1)
        return d_hat

    def _objective_with_loss_ratio(self):
        def loss_ratio_objective(**kwargs):
            acc, mean_process_time, var_process_time = self._black_box_function(**kwargs)
            d_hat = self._estimate_loss_ratio(mean_process_time, var_process_time)

            # square difference between the estimated loss ratio and the preset loss ratio
            norm = np.max(d_hat - self.preset_loss_ratio, 0)
            # norm = np.square(d_hat - self.preset_loss_ratio)

            # maximize accuracy with loss ratio constrain which represent with Lagrange multiplier
            objective_function = acc - self.penalty * norm
            if np.isnan(objective_function):
                return -10000
            return objective_function

        return loss_ratio_objective

    def _objective_loss_ratio_with_regularize(self):
        def loss_ratio_with_regularize(**kwargs):
            thresholds = np.array(list(kwargs.values()))
            acc, mean_process_time, var_process_time = self._black_box_function(**kwargs)
            d_hat = self._estimate_loss_ratio(mean_process_time, var_process_time)

            # square difference between the estimated loss ratio and the preset loss ratio
            norm = np.max(d_hat - self.preset_loss_ratio, 0)

            # push to 1
            regularize = np.sum(np.square(self.max_threshold - thresholds))
            objective_function = acc - self.penalty * norm - self.weight_decay * regularize

            if np.isnan(objective_function):
                return -10000
            return objective_function

        return loss_ratio_with_regularize

    def _objective_loss_ratio_with_regularize_two_side(self):
        def loss_ratio_with_regularize(**kwargs):
            thresholds = np.array(list(kwargs.values()))
            acc, mean_process_time, var_process_time = self._black_box_function(**kwargs)
            d_hat = self._estimate_loss_ratio(mean_process_time, var_process_time)

            # square difference between the estimated loss ratio and the preset loss ratio
            norm = np.max(d_hat - self.preset_loss_ratio, 0)

            # push to two side
            regularize = np.sum((self.max_threshold - thresholds) * (thresholds - self.mini_threshold))
            objective_function = acc - self.penalty * norm - self.weight_decay * regularize
            if np.isnan(objective_function):
                return -10000
            return objective_function

        return loss_ratio_with_regularize
