from WPGAN_models import Discriminator, Generator
from config import Config
from utils import size_to_depth, depth_to_size
import torch
import os
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision import datasets

print("CUDA_HOME :{}".format(os.environ.get('CUDA_HOME')))
print("CNN_HOME  :{}".format(os.environ.get('CNN_HOME')))
print("GPUs: {}".format(Config.ngpus))


class Trainer:
    def __init__(self, project_name, batch_scale):
        self.project_name = project_name
        self.project_param = Config.Project_Params[project_name]

        self.start_depth = self.project_param['start_depth']
        self.end_depth = self.project_param['end_depth']
        self.data_dir = self.project_param['data_dir']
        self.alpha_list = self.project_param['alpha_list']
        self.delta_alpha = self.project_param['delta_alpha']
        self.epos_list = self.project_param['epos_list']
        self.batch_list = self.project_param['batch_list']
        self.save_internal = self.project_param['save_internal']
        self.noise_net = self.project_param['Noise_Net']
        self.resnet = self.project_param['Resnet']
        self.nz_dim = 256
        self.latent_dim = self.project_param['latent_dim']
        self.batch_scale = float(batch_scale)

        print("Dataset Folder: {}".format(self.data_dir))

        if not os.path.isdir(Config.model_root):
            os.mkdir(Config.model_root)

        self.model_dir = os.path.join(Config.model_root, self.project_name)
        project_exist = True
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
            project_exist = False

        print("Project Exist: {}".format(project_exist))
        G_net = Generator(self.nz_dim, self.latent_dim, 1024, noise_net=self.noise_net, resnet=self.resnet).to(Config.devices[0])
        self.G_net = nn.DataParallel(G_net, device_ids=Config.device_groups[0])
        D_net = Discriminator(self.latent_dim, 1024, resnet=self.resnet).to(Config.devices[1])
        self.D_net = nn.DataParallel(D_net, device_ids=Config.device_groups[1])

        # start_depth = model_state_dict['current_depth']
        self.G_optim = optim.Adam(self.G_net.parameters(), lr=self.project_param['G_lr'], betas=(0, 0.99))
        self.D_optim = optim.Adam(self.D_net.parameters(), lr=self.project_param['D_lr'], betas=(0, 0.99))

        torch.set_default_tensor_type(torch.FloatTensor)

    def load_module(self):
        pretrained_file = os.path.join(self.model_dir, "model_{}.pth".format(self.project_param['load_depth']))
        if os.path.isfile(pretrained_file):
            model_state_dict = torch.load(pretrained_file, map_location=torch.device(Config.devices[0]))
            self.G_net.load_state_dict(model_state_dict['G_net'], strict=True)
            self.D_net.load_state_dict(model_state_dict['D_net'], strict=True)

            print("Loaded model file: {}".format(pretrained_file))
            if self.project_param['Use_last_alpha']:
                self.project_param['alpha_list'][model_state_dict['current_depth']] = model_state_dict['alpha']

            del model_state_dict

    def dump_model(self, depth, step):
        save_dict = {'G_net': self.G_net.state_dict(), 'D_net': self.D_net.state_dict(), 'current_depth': depth,
                     'noise_size': 256, 'latent_size': self.latent_dim, 'alpha': self.G_net.module.get_alpha()}
        save_path = os.path.join(self.model_dir, "model_{}.pth".format(depth))
        torch.save(save_dict, save_path)

        if step % 30000 == 0:
            save_path = os.path.join(self.model_dir, "model_{}_{}.pth".format(depth, step))
            torch.save(save_dict, save_path)

    def start_training(self):
        print("start training...")
        for depth in range(self.start_depth, self.end_depth + 1):
            batch_size = int(self.batch_list[depth] * self.batch_scale)
            img_size = depth_to_size(depth)
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = datasets.ImageFolder(self.data_dir, transform=transform)
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=16)

            self.G_net.module.set_depth(depth, alpha_start=self.alpha_list[depth], delta_alpha=self.delta_alpha)
            self.D_net.module.set_depth(depth, alpha_start=self.alpha_list[depth], delta_alpha=self.delta_alpha)

            print("Batch Size: {}".format(batch_size))
            step = 0
            for epo in range(self.epos_list[depth]):
                for batch_ndx, sample in enumerate(data_loader):
                    step += 1
                    sample = sample[0].to(Config.devices[1])

                    self.D_optim.zero_grad()
                    if self.noise_net:
                        noise = torch.randn(sample.shape[0], self.nz_dim).to(Config.devices[0])
                    else:
                        noise = torch.randn(sample.shape[0], self.latent_dim).to(Config.devices[0])

                    fake = self.G_net(noise).to(Config.devices[1])
                    fake_out = self.D_net(fake.detach())
                    real_out = self.D_net(sample)

                    # update D
                    # Gradient Penalty
                    eps = torch.rand(sample.shape[0], 1, 1, 1).to(Config.devices[1])
                    eps = eps.expand_as(sample)
                    x_hat = eps * sample + (1 - eps) * fake.detach()
                    x_hat.requires_grad = True
                    px_hat = self.D_net(x_hat)
                    grad = torch.autograd.grad(
                        outputs=px_hat.sum(),
                        inputs=x_hat,
                        create_graph=True
                    )[0]
                    grad_norm = grad.view(sample.shape[0], -1).norm(2, dim=1)
                    gradient_penalty = 10 * ((grad_norm - 1) ** 2).mean()

                    # Wasserstein distance
                    D_loss = fake_out.mean() - real_out.mean() + gradient_penalty
                    D_loss.backward()
                    if self.project_param['train_last_layer_only']:
                        self.D_net.module.scale_grad(0)

                    self.D_optim.step()

                    # update G
                    self.G_optim.zero_grad()
                    fake_out = self.D_net(fake)

                    G_loss = -fake_out.mean()
                    G_loss.backward()
                    if self.project_param['train_last_layer_only']:
                        self.G_net.module.scale_grad(0)

                    self.G_optim.step()

                    print("\r Step: {}; G_loss: {:.4f}; D_loss: {:.4f}; Alpha: [{:.3f}, {:.3f}]".format(step, G_loss.mean(),
                            D_loss.mean().item(), self.G_net.module.get_alpha(), self.D_net.module.get_alpha()), end="")

                    if step % self.save_internal[depth] == 0:
                        self.dump_model(depth, step)
                        print("\n Model saves at Depth: {}; Step: {}".format(depth, step))

                    if step % 20 == 0:
                        self.G_net.module.increase_alpha()
                        self.D_net.module.increase_alpha()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-b', '--batch_scale', default=1, required=False)
    args = parser.parse_args()
    project_name = args.name
    batch_scale = args.batch_scale
    trainer = Trainer(project_name, batch_scale)
    trainer.load_module()
    trainer.start_training()