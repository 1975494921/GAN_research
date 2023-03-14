from torchvision.utils import make_grid, save_image
import torch
from torch import nn
import os
from WPGAN_models import Generator, Discriminator
from utils import size_to_depth
from config import Config
import math
import numpy as np

torch.manual_seed(0)
num_generate = 16
num_grid = 9
img_size = 512
model_root = 'model_trains'
model_key = "anime_project1"
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

latent_dim = Config.Project_Params[model_key]['latent_dim']

G_net = Generator(256, latent_dim, 1024).to(device)
G_net = nn.DataParallel(G_net, device_ids=device_ids)
G_net.module.set_depth(model_depth, alpha_start=1)

D_net = Discriminator(256, 1024).to(device)
D_net = nn.DataParallel(D_net, device_ids=device_ids)
D_net.module.set_depth(model_depth, alpha_start=1)

model_state_dict = torch.load(model_file, map_location=device)
G_net.load_state_dict(model_state_dict['G_net'])
G_net.eval()

D_net.load_state_dict(model_state_dict['D_net'])
D_net.eval()

with torch.no_grad():
    for i in range(num_generate):
        noise = torch.randn(num_grid * 2, latent_dim).to(device)
        fake = G_net(noise)
        fake_scores = D_net(fake).cpu().numpy().reshape(-1)
        indices = np.argsort(-fake_scores)
        indices = indices[:num_grid]
        indices.sort()
        print(indices)
        fake = fake[indices]
        img_path = os.path.join(image_dir, "image_{}.jpg".format(i))
        save_image(make_grid(fake, padding=1, normalize=True, nrow=int(math.sqrt(num_grid))), img_path)
        print("Save image: {}".format("image_{}.jpg".format(i)))
