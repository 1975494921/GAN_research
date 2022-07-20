import torch
from PGAN_utils import size_to_depth, depth_to_size


class Config:
    debug = False
    ngpus = min(torch.cuda.device_count(), 8)
    gpu_shift = 2
    device_groups = [[i for i in range(gpu_shift, ngpus + gpu_shift)],
                     [i for i in range(gpu_shift, ngpus + gpu_shift)]]

    devices = ["cuda:{}".format(item[0]) for item in device_groups]
    model_root = 'model_trains'

    Project_Params = dict()
    Project_Params['cartoon_train001'] = {
        'load_depth': 5,
        'start_depth': 5,
        'end_depth': size_to_depth(1024),

        'Use_last_alpha': True,
        'Use_Mean': True,
        'G_lr': 0.0001,
        'D_lr': 0.00005,

        'data_dir': '/home/zceelil/dataset/cartoon/imsize_500/cartoonset100k',
        'epos_list': [0, 300, 300, 480, 60, 60, 60, 60, 40, 40],
        'batch_list': [0, 1000, 1000, 1000, 1000, 580, 200, 100, 80, 50],
        'save_internal': [0, 20, 20, 20, 20, 5, 2, 1, 1, 1],
        'alpha_list': [0, 0.01, 0.01, 0.01, 0.01, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00],
        'delta_alpha': 0.002,

        'train_last_layer_only': False,
    }
