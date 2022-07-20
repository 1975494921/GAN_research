import torch


class Config:
    debug = False

    ngpus = min(torch.cuda.device_count(), 8)
    gpu_shift = 0
    device_groups = [[i for i in range(gpu_shift, ngpus + gpu_shift)],
                     [i for i in range(gpu_shift, ngpus + gpu_shift)]]

    devices = ["cuda:{}".format(item[0]) for item in device_groups]

    nz_dim = 32  # Noise dim

    nz_side = 1

    batch_size = 256
