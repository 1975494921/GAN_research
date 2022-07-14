
# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


class Config:
    debug = False

    device_groups = [[0, 1, 2, 3], [2, 3, 0, 1]]
    devices = ["cuda:{}".format(item[0]) for item in device_groups]

    text_embedding_dim = 512
    text_encoder = "distiluse-base-multilingual-cased-v1"

    nz_dim = 32  # Noise dim

    nz_side = 1

    nz_num = nz_dim * nz_side * nz_side  # (nz_dim, nz_side_length, nz_side_length)

    batch_size = 256

    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64