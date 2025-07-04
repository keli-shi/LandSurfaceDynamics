import numpy as np
from scipy import stats
import os
import sys
import yaml
from munch import munchify

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)


def calculate_intrinsic_dimension_statistics(dataset, model_type='model', if_image=False):
    configs_dir = os.path.join('configs', dataset, model_type)
    print(configs_dir)
    if if_image:
        filename = 'intrinsic_dimension_image.npy'
    else:
        filename = 'intrinsic_dimension.npy'
    dims_all = []
    for config_filepath in os.listdir(configs_dir):
        cfg = load_config(filepath=os.path.join(configs_dir, config_filepath))
        cfg = munchify(cfg)
        log_dir = '_'.join([cfg.log_dir, cfg.dataset, cfg.model_name, str(cfg.seed)])
        vars_filepath = os.path.join(log_dir, 'variables')
        dims = np.load(os.path.join(vars_filepath, filename))
        dims_all.append(dims)
    dims_all = np.concatenate(dims_all)
    dim_mean = np.mean(dims_all)
    dim_std = np.std(dims_all)
    print('Mean (±std):', '%.2f (±%.2f)' % (dim_mean, dim_std))
    print('Confidence interval:', '(%.1f, %.1f)' % (dim_mean-1.96*dim_std, dim_mean+1.96*dim_std))




if __name__ == '__main__':
    dataset = str(sys.argv[1])
    calculate_intrinsic_dimension_statistics(dataset, 'model64')