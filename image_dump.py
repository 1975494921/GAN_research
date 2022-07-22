from torchvision.utils import make_grid, save_image
import torch
from torch import nn
import os
from WPGAN_models import Generator, Discriminator
from PGAN_utils import size_to_depth
import math

num_generate = 32
num_grid = 16
img_size = 64
model_root = 'model_trains'
model_key = "cartoon_train002"
model_dir = os.path.join(model_root, model_key)

device_ids = [0, 1]
device = torch.device("cuda:{}".format(device_ids[0]))

image_root = "image_results"
if not os.path.isdir(image_root):
    os.mkdir(image_root)

model_depth = size_to_depth(img_size)
model_file = os.path.join(model_dir, "model_{}.pth".format(model_depth))
if not os.path.isfile(model_file):
    raise FileNotFoundError("{} not existed".format(model_file))

print("Use model file: {}".format(model_file))

image_dir = os.path.join(image_root, model_key)
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

image_dir = os.path.join(image_dir, "Depth_{}".format(model_depth))
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)


G_net = Generator(256, 512, 1024).to(device)
G_net = nn.DataParallel(G_net, device_ids=device_ids)
G_net.module.set_depth(model_depth, alpha_start=1)

D_net = Discriminator(512, 1024).to(device)
D_net = nn.DataParallel(D_net, device_ids=device_ids)
D_net.module.set_depth(model_depth, alpha_start=1)

model_state_dict = torch.load(model_file, map_location=device)
G_net.load_state_dict(model_state_dict['G_net'])
G_net.eval()
D_net.load_state_dict(model_state_dict['D_net'])
D_net.eval()

# D_net = Discriminator(512, 1024).to(device)
# D_net = nn.DataParallel(D_net, device_ids=device_ids)
# D_net.module.set_depth(model_depth, alpha_start=1)
# D_net.load_state_dict(model_state_dict['D_net'])
# D_net.eval()

for i in range(num_generate):
    noise = torch.randn(num_grid, 512).to(device)
    fake = G_net(noise)
    scores = D_net(fake)
    print(scores)

    img_path = os.path.join(image_dir, "image_{}.jpg".format(i))
    save_image(make_grid(fake, padding=1, normalize=True, nrow=4), img_path)
    print("Save image: {}".format("image_{}.jpg".format(i)))
