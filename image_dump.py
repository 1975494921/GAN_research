from torchvision.utils import make_grid, save_image
import torch
from torch import nn
import os
from PGAN_models import Generator, Discriminator
from PGAN_utils import size_to_depth
import math
import time

num_generate = 16
img_size = 64
model_root = 'model_trains'
model_key = "train001"
model_dir = os.path.join(model_root, model_key)
generate_interval = 300

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


G_net = Generator(256, 512, 1024).to(device)
G_net = nn.DataParallel(G_net, device_ids=device_ids)
G_net.module.set_depth(model_depth, alpha_start=1)

while True:
    model_state_dict = torch.load(model_file, map_location=device)
    G_net.load_state_dict(model_state_dict['G_net'])
    del model_state_dict

    with torch.no_grad():
        noise = torch.randn(num_generate, 256).to(device)
        fake = G_net(noise)
        img_path = os.path.join(image_dir, "image_{}.jpg".format(model_depth))
        save_image(make_grid(fake, padding=1, normalize=True, nrow=int(math.sqrt(num_generate))), img_path)
        t = time.localtime()
        print("Save image at {:02d}:{:02d} -- {}".format(t.tm_hour, t.tm_min, os.path.basename(img_path)))

    time.sleep(generate_interval)
