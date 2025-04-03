import json
import re

import numpy as np


def get_machine_type(hostname: str):
    if "worker" in hostname:
        return "workers"
    elif "edge" in hostname:
        return "edges"
    return "clouds"


def process_time_profile(system_profile, gates_layer_idx, activated_gates_idx=None, start_layer=None, end_layer=None,
                         sample_size=5000, network_config=None, normalize=True):
    """
    :param system_profile:
    :param gates_layer_idx: the idx of gate layers in the model
    :param activated_gates_idx: should be the idx of activated gates in the gates_layer_idx list
    :param start_layer: the start layer of the profile
    :param end_layer: the end layer of the profile
    :param sample_size: the number of samples to generate
    :param network_config: the network configuration
    :param normalize:can be bool or int or float. If bool then normalize by the maximum process time given the
    activated gates. If int or float then normalize by the given value.
    :return:
    """
    # load system profile
    if isinstance(system_profile, str):
        with open(system_profile, 'r', encoding='utf8') as f:
            system_profile = json.load(f)

    if activated_gates_idx:
        activated_gates = [gates_layer_idx[i] for i in activated_gates_idx]
        activated_gates.append(gates_layer_idx[-1])  # make sure the final gates is always activated
        activated_gates = list(set(activated_gates))
    else:
        activated_gates = gates_layer_idx

    result_dict = dict()
    sample_x = np.zeros(sample_size)
    backbones_process_time = system_profile['backbone_execution_time']
    gate_process_time = system_profile['gate_execution_time']

    start_layer = start_layer if start_layer else 0
    end_layer = end_layer - 1 if end_layer else len(backbones_process_time.keys()) - 1

    # search over backbone execution time
    for i, (b_layer_name, b_layer_value) in enumerate(backbones_process_time.items()):
        if i < start_layer:
            # skip layers before start layer
            continue

        sample_x = sample_x + np.random.lognormal(mean=b_layer_value['log_mean'],
                                                  sigma=b_layer_value['log_std'], size=sample_size)

        # extract layer number from layer name
        layer_id = extract_layer_number(b_layer_name)
        if layer_id in activated_gates:
            # if layer is a gate layer search over gate execution time
            for g_layer_name, g_layer_value in gate_process_time.items():
                if f'layer_{layer_id}' in g_layer_name:
                    # if layer is in the current gate layer add

                    sample_x = sample_x + np.random.lognormal(mean=g_layer_value['log_mean'],
                                                              sigma=g_layer_value['log_std'],
                                                              size=sample_size)

            result_dict[gates_layer_idx.index(layer_id)] = {'mean': sample_x.mean(),
                                                            'var': sample_x.var(),
                                                            'std': sample_x.std(),
                                                            'log_mean': np.mean(np.log(sample_x)),
                                                            'log_std': np.std(np.log(sample_x))}
        if i == end_layer:
            # -1 special for transmit layer
            if network_config:
                sample_x = sample_x + np.random.lognormal(mean=network_config['log_mean'],
                                                          sigma=network_config['log_std'],
                                                          size=sample_size)

            result_dict[-1] = {'mean': sample_x.mean(),
                               'std': sample_x.std(),
                               'var': sample_x.var(),
                               'log_mean': np.mean(np.log(sample_x)),
                               'log_std': np.std(np.log(sample_x))}
            break

    if normalize:
        if isinstance(normalize, bool):
            max_process_time = max(result_dict.values(), key=lambda x: x['mean'])['mean']
        elif isinstance(normalize, int) or isinstance(normalize, float):
            max_process_time = normalize
        else:
            raise ValueError(f"invalid normalize value {normalize}")
        result_dict = {k: {'mean': v['mean'] / max_process_time, 'var': v['var'] / max_process_time ** 2}
                       for k, v in result_dict.items()}
    return result_dict


def extract_layer_number(s):
    match = re.search(r"layer_(\d+)", s)
    return int(match.group(1)) if match else -1


def get_maximum_process_time(system_profile):
    if isinstance(system_profile, str):
        with open(system_profile, 'r', encoding='utf8') as f:
            system_profile = json.load(f)

    cumulate_time = 0
    backbones_process_time = system_profile['backbone_execution_time']
    gate_process_time = system_profile['gate_execution_time']
    cumulate_time += sum(item["mean"] for item in backbones_process_time.values())
    cumulate_time += sum(item["mean"] for item in gate_process_time.values())
    return cumulate_time
