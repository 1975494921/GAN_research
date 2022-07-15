class Config:
    debug = False

    device_groups = [[2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]]
    devices = ["cuda:{}".format(item[0]) for item in device_groups]

    nz_dim = 32  # Noise dim

    nz_side = 1

    batch_size = 256
