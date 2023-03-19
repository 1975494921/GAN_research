from torchvision.utils import make_grid, save_image
import torch
from torch import nn
import os
from models import Generator, Discriminator
from utils import size_to_depth
from config import Config
import math
import numpy as np
import argparse

device_ids = [0, 1]
device = torch.device("cuda:{}".format(device_ids[0]))
torch.manual_seed(0)

model_root = 'model_trains'
image_root = 'image_results'


def main(model_key, img_size, num_generate, grid_size, select_probability):
    model_dir = os.path.join(model_root, model_key)
    num_grid = int(grid_size * grid_size)

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

    generate_num = int(num_grid / select_probability)

    with torch.no_grad():
        for i in range(num_generate):
            noise = torch.randn(generate_num, latent_dim).to(device)
            fake = G_net(noise)
            fake_scores = D_net(fake).cpu().numpy().reshape(-1)
            indices = np.argsort(-fake_scores)
            indices = indices[:num_grid]
            indices.sort()
            fake = fake[indices]
            img_path = os.path.join(image_dir, "image_{}.jpg".format(i))
            save_image(make_grid(fake, padding=1, normalize=True, nrow=int(math.sqrt(num_grid))), img_path)
            print("Save image: {}".format("image_{}.jpg".format(i)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--image_size', default=512, required=False)
    parser.add_argument('--num_image', default=16, required=False)
    parser.add_argument('--grid_size', default=8, required=False)
    parser.add_argument('--select_probability', default=0.5, required=False)
    args = parser.parse_args()
    project_name = args.name
    image_size = int(args.image_size)
    num_generate = int(args.num_image)
    grid_size = int(args.grid_size)
    select_probability = float(args.select_probability)
    assert image_size in [64, 128, 256, 512, 1024], "Image size must be 64, 128, 256, 512 or 1024"
    assert num_generate > 0, "Number of images must be greater than 0"
    assert grid_size > 0, "Grid size must be greater than 0"
    assert 0 < select_probability <= 1, "Select probability must be in (0, 1]"
    main(project_name, image_size, num_generate, grid_size, select_probability)
