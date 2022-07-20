from PGAN_models import Discriminator, Generator
from config import Config
from PGAN_utils import size_to_depth, depth_to_size
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
    def __init__(self, project_name):
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

        print("Dataset Folder: {}".format(self.data_dir))

        if not os.path.isdir(Config.model_root):
            os.mkdir(Config.model_root)

        self.model_dir = os.path.join(Config.model_root, self.project_name)
        project_exist = True
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
            project_exist = False

        print("Project Exist: {}".format(project_exist))
        G_net = Generator(256, 512, 1024).to(Config.devices[0])
        self.G_net = nn.DataParallel(G_net, device_ids=Config.device_groups[0])
        D_net = Discriminator(512, 1024).to(Config.devices[1])
        self.D_net = nn.DataParallel(D_net, device_ids=Config.device_groups[1])

        # start_depth = model_state_dict['current_depth']
        self.G_optim = optim.Adam(self.G_net.parameters(), lr=self.project_param['G_lr'])
        self.D_optim = optim.Adam(self.D_net.parameters(), lr=self.project_param['D_lr'])
        self.loss_func = nn.BCELoss()
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

        else:
            raise FileNotFoundError("Model file: {} not found".format(pretrained_file))

    def dump_model(self, depth):
        save_dict = {'G_net': self.G_net.state_dict(), 'D_net': self.D_net.state_dict(), 'current_depth': depth,
                     'noise_size': 256, 'latent_size': 512, 'alpha': self.G_net.module.get_alpha()}
        save_path = os.path.join(self.model_dir, "model_{}.pth".format(depth))
        torch.save(save_dict, save_path)

    def start_training(self):
        print("start training...")
        for depth in range(self.start_depth, self.end_depth + 1):
            D_Use_Mean = True if self.project_param['Use_Mean'] else False
            batch_size = self.batch_list[depth]
            img_size = depth_to_size(depth)
            transform = transforms.Compose([
                transforms.Resize(max(img_size, 128)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = datasets.ImageFolder(self.data_dir, transform=transform)
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=16)

            self.G_net.module.set_depth(depth, alpha_start=self.alpha_list[depth], delta_alpha=self.delta_alpha)
            self.D_net.module.set_depth(depth, alpha_start=self.alpha_list[depth], delta_alpha=self.delta_alpha)

            for epo in range(1, self.epos_list[depth] + 1):
                if epo > (self.epos_list[depth] // 3 * 2):
                    D_Use_Mean = False

                for batch_ndx, sample in enumerate(data_loader):
                    sample = sample[0].to(Config.devices[1])
                    real_label = torch.full((sample.shape[0], 1,), 1, dtype=torch.float).to(Config.devices[1])
                    fake_label = torch.full((sample.shape[0], 1,), 0, dtype=torch.float).to(Config.devices[1])

                    # update D
                    self.D_optim.zero_grad()
                    noise = torch.randn(sample.shape[0], 256).to(Config.devices[0])
                    fake = self.G_net(noise).to(Config.devices[1])
                    fake_out = self.D_net(fake.detach())
                    real_out = self.D_net(sample)

                    # gradient_penalty = 0
                    # if False:
                    #     ## Gradient Penalty
                    #     eps = torch.rand(sample.shape[0], 1, 1, 1).to(Config.devices[1])
                    #     eps = eps.expand_as(sample)
                    #     x_hat = eps * sample + (1 - eps) * fake.detach()
                    #     x_hat.requires_grad = True
                    #     px_hat = D_net(x_hat)
                    #     grad = torch.autograd.grad(
                    #         outputs=px_hat.sum(),
                    #         inputs=x_hat,
                    #         create_graph=True
                    #     )[0]
                    #     grad_norm = grad.view(sample.shape[0], -1).norm(2, dim=1)
                    #     gradient_penalty = 2 * ((grad_norm - 1) ** 2).mean()

                    ###########

                    D_loss_real = self.loss_func(real_out, real_label)
                    D_loss_fake = self.loss_func(fake_out, fake_label)

                    if D_Use_Mean:
                        D_loss = D_loss_real.mean() + D_loss_fake.mean()
                    else:
                        D_loss = D_loss_real + D_loss_fake

                    D_loss.backward()
                    if self.project_param['train_last_layer_only']:
                        self.D_net.module.scale_grad(0)

                    self.D_optim.step()

                    ## update G
                    self.G_optim.zero_grad()
                    fake_out = self.D_net(fake)

                    G_loss = self.loss_func(fake_out, real_label)
                    G_loss.backward()
                    if self.project_param['train_last_layer_only']:
                        self.G_net.module.scale_grad(0)
                        self.G_net.module.scale_grad_noise(0)

                    self.G_optim.step()

                    ##############
                    print("\r Epo: {}; G_loss: {:.4f}; D_loss: {:.4f}; Alpha: [{:.3f}, {:.3f}]".format(epo, G_loss.mean(),
                            D_loss.mean().item(), self.G_net.module.get_alpha(), self.D_net.module.get_alpha()), end="")

                if epo % self.save_internal[depth] == 0:
                    self.dump_model(depth)
                    print("\n Model saves at Depth: {}; Epoch: {}".format(depth, epo))

                self.G_net.module.increase_alpha()
                self.D_net.module.increase_alpha()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    args = parser.parse_args()
    project_name = args.name
    trainer = Trainer(project_name)
    trainer.load_module()
    trainer.start_training()

