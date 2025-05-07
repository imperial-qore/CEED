import os
import argparse
import pickle

from tqdm import tqdm

try:
    from .bo_threshold_tunner import BOThresholdTunner
    from .constants import *
except:
    import sys

    sys.path.append("../../../../")
    from app.app_src.scheduler.bo_offline_scheduler.bo_threshold_tunner import BOThresholdTunner
    from app.app_src.scheduler.bo_offline_scheduler.constants import *


def main(hp):
    os.makedirs(hp.cache_dir, exist_ok=True)

    system_profile = os.path.join(hp.model_path, hp.machine_type, "profile_summary.json")
    threshold_tunner = BOThresholdTunner(system_profile=system_profile, **vars(hp))

    lookup_table = dict()

    # whether to reload the data set
    if threshold_tunner.reload_cache_folder():
        dataloader = threshold_tunner.load_data(hp.dataset_cache_dir,
                                                hp.dataset,
                                                hp.batch_size,
                                                split=hp.split,
                                                valid_mode=hp.valid_mode,
                                                random_seed=hp.seed,
                                                ratio=hp.valid_ratio)
        threshold_tunner.prepare_prediction(dataloader)

    for arrival_rate in tqdm(ARRIVAL_RATES):
        if not hp.int:
            arrival_rate = 1 / arrival_rate
        threshold_tunner.arrival_rate = arrival_rate
        optimize_result = threshold_tunner.threshold_tuning(util_kind=hp.util_kind,
                                                            init_points=hp.init_points,
                                                            n_iters=hp.n_iters,
                                                            early_stop_count=hp.early_stop_count,
                                                            seed=hp.seed,
                                                            save=False,
                                                            process_verbose=hp.process_verbose)

        threshold = list(optimize_result['params'].values())
        lookup_table[1/arrival_rate] = threshold

    save_folder = os.path.join(hp.save_folder, hp.machine_type)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "bo_lookup_table.pkl"), "wb") as f:
        pickle.dump(lookup_table, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--model_path',
                        default="models/resnet50_bn_small_ee_cifar_10_end_join",
                        type=str)
    parser.add_argument("--machine_type", default="worker", type=str)

    parser.add_argument("--dataset", default="cifar_10", type=str)
    parser.add_argument("--dataset_cache_dir",
                        default="source/datasets",
                        type=str)
    parser.add_argument('--cache_dir',
                        default="../../../../cache/{dataset}",
                        type=str)
    parser.add_argument('--batch_size', default=5000, type=int)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--valid_mode", default="valid", type=str)
    parser.add_argument("--valid_ratio", default=0.5, type=float)

    parser.add_argument('--mini_threshold', default=0.0, type=float)
    parser.add_argument('--max_threshold', default=1.01, type=float)
    parser.add_argument('--util_kind', default="ucb", type=str)
    parser.add_argument("--init_points", default=5, type=int)
    parser.add_argument("--n_iters", default=100, type=int)
    parser.add_argument("--early_stop_count", default=None, type=int)

    parser.add_argument("--objective", default="loss_ratio", type=str)
    parser.add_argument('--max_queue_length', default=10, type=int)
    parser.add_argument("--loss_ratio", default=0, type=float)
    parser.add_argument("--penalty", default=100, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)

    parser.add_argument("--save_folder", default="source/", type=str)
    parser.add_argument("--process_verbose", type=int, default=2,
                        help="0: no verbose, 1: short verbose, 2: full verbose")
    parser.add_argument("--int", default=False, action="store_true")
    hp = parser.parse_args()
    hp.cache_dir = hp.cache_dir.format(dataset=hp.dataset)

    main(hp)
