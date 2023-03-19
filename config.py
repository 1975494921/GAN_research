import torch
from utils import size_to_depth, depth_to_size


class Config:
    debug = False
    ngpus = min(torch.cuda.device_count(), 8)
    gpu_shift = 0
    gpu_shift = min(gpu_shift, torch.cuda.device_count() - ngpus)
    device_groups = [[i for i in range(gpu_shift, ngpus + gpu_shift)],
                     [i for i in range(gpu_shift, ngpus + gpu_shift)]]

    devices = ["cuda:{}".format(item[0]) for item in device_groups]
    model_root = 'model_trains'

    Project_Params = dict()

    Project_Params['landscape_train001'] = {
        'load_depth': 6,
        'start_depth': 6,
        'end_depth': size_to_depth(256),

        'Use_last_alpha': True,
        'Use_Mean': True,
        'Noise_Net': False,
        'Resnet': False,
        'G_lr': 0.0002,
        'D_lr': 0.0001,
        'latent_dim': 512,

        'data_dir': '/scratch/zceelil/dataset/landscape/imsize_256',
        'epos_list': [0, 1000, 600, 480, 400, 900, 900, 900, 900, 900],
        'batch_list': [0, 50, 50, 50, 50, 50, 50, 50, 50, 50],
        'save_internal': [10000, 10000, 10000, 10000, 2000, 1000, 1000, 1000, 1000, 1000],
        'alpha_list': [0, 0.01, 0.01, 0.01, 0.01, 1, 0.00, 0.00, 0.00, 0.00, 0.00],
        'delta_alpha': 0.001,

        'train_last_layer_only': False,
    }

    Project_Params['anime_project1'] = {
        'load_depth': 8,
        'start_depth': 8,
        'end_depth': size_to_depth(512),

        'Use_last_alpha': True,
        'Use_Mean': True,
        'Noise_Net': False,
        'Resnet': False,
        'G_lr': 0.0001,
        'D_lr': 0.0001,
        'latent_dim': 256,

        'data_dir': '/scratch/zceelil/dataset/cartoon/img_512',
        'epos_list': [0, 1000, 100, 100, 100, 100, 100, 100, 100, 100],
        'batch_list': [0, 50, 50, 50, 50, 50, 30, 40, 40, 40],
        'save_internal': [10000, 10000, 10000, 5000, 5000, 5000, 5000, 5000, 5000, 2000],
        'alpha_list': [0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        'delta_alpha': 0.0001,

        'train_last_layer_only': False,
    }

    Project_Params['portrait_project1'] = {
        'load_depth': 7,
        'start_depth': 7,
        'end_depth': size_to_depth(512),

        'Use_last_alpha': True,
        'Use_Mean': True,
        'Noise_Net': False,
        'Resnet': False,
        'G_lr': 0.0002,
        'D_lr': 0.0002,
        'latent_dim': 256,

        'data_dir': '/scratch/zceelil/dataset/portrait',
        'epos_list': [0, 1000, 100, 100, 100, 10000, 10000, 10000, 10000, 10000],
        'batch_list': [0, 50, 50, 50, 40, 80, 10, 10, 10, 40],
        'save_internal': [10000, 10000, 10000, 2000, 2000, 5000, 5000, 5000, 5000, 5000],
        'alpha_list': [0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        'delta_alpha': 0.0002,

        'train_last_layer_only': False,
    }
