# GAN_research - Junting Li

University College London, Department of Electrical and Electronic Engineering, 2023

This repository contains the 3rd year project codes from Junting Li.

To set up the project in your python coding environment (IDE):

1. Clone the repository with git with the following command:
   ```commandline
   git clone https://github.com/1975494921/GAN_research
   ```
2. Add a virtual environment (venv) and install the requirements from requirements.txt:
    ```commandline
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3. Modify the config.py to add your project configuration. The following is the sample:

        Project_Params['project_name'] = {
           'load_depth': 8,
           'start_depth': 8,
           'end_depth': size_to_depth(512),

           'Use_last_alpha': True,
           'Use_Mean': True,
           'Noise_Net': False,
           'Resnet': False,
           'G_lr': 0.0001,
           'D_lr': 0.0001,
           'latent_dim': 256,

           'data_dir': '/scratch/zceelil/dataset/cartoon/img_512',
           'epos_list': [0, 1000, 100, 100, 100, 100, 100, 100, 100, 100],
           'batch_list': [0, 50, 50, 50, 50, 50, 30, 40, 40, 40],
           'save_internal': [10000, 10000, 10000, 5000, 5000, 5000, 5000, 5000, 5000, 2000],
           'alpha_list': [0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
           'delta_alpha': 0.0001,

           'train_last_layer_only': False,
        }
   The load_depth is the depth of the model to be loaded. The start_depth is the depth of the model to be trained. The end_depth is the depth of the model to be trained to. The Use_last_alpha is a boolean value to indicate whether to use the last alpha value in the alpha_list. The Use_Mean is a boolean value to indicate whether to use the mean of the alpha_list. The Noise_Net is a boolean value to indicate whether to use the noise net. The Resnet is a boolean value to indicate whether to use the resnet. The G_lr is the learning rate of the generator. The D_lr is the learning rate of the discriminator. The latent_dim is the latent dimension of the noise. The data_dir is the directory of the dataset. The epos_list is the list of the epochs for each depth. The batch_list is the list of the batch size for each depth. The save_internal is the list of the save interval for each depth. The alpha_list is the list contain the alpha value which control the fade in for each depth. The delta_alpha is the delta alpha value at a given time step for each depth. The train_last_layer_only is a boolean value to indicate whether to train the last layer only.


4. Start to train the model with the following command:
    ```commandline
    python3 train.py --name project_name
    ```
   The project is the name of the project in the config.py. Note that the GPU is required to train the model. You can scale the batch size to fit your GPU memory with --batch_scale argument. For example, if you have a GPU with 16GB memory, you can scale the batch size to 8 with the following command:
    ```commandline
    python3 train.py --name project_name --batch_scale 8
    ```


5. Start to use the model with the following command:
    ```commandline
    python3 image_dump.py --name project_name
    ```
   To use this code, you will need to pass several command-line arguments when running the script:

   -n or --name: (required) the name of the project, which should correspond to a pre-trained GAN model that is available in the model_trains directory. For example, if you have a pre-trained GAN model for generating faces called faces.pth, you would pass -n faces to use this model.
   
   -s or --image_size: (optional, default=512) the size of the generated images in pixels. This should be one of 64, 128, 256, 512, or 1024.
   
   --num_image: (optional, default=16) the number of images to generate.

   --grid_size: (optional, default=8) the size of the grid to arrange the generated images. For example, if --grid_size is 8, the output grid will have 64 images arranged in an 8x8 grid.

   --select_probability: (optional, default=0.5) the probability of selecting each generated image for the output grid. For example, if --select_probability is 0.5, approximately half of the generated images will be included in the output grid.


# References

[1] EmilienDupont. (n.d.). EmilienDupont/WGAN-GP: Pytorch implementation of Wasserstein Gans with gradient penalty. GitHub. Retrieved March 19, 2023, from https://github.com/EmilienDupont/wgan-gp 

[2] Ziwei-Jiang. (n.d.). Ziwei-Jiang/PGGAN-pytorch: A pytorch implementation of progressive growing gan. GitHub. Retrieved March 19, 2023, from https://github.com/ziwei-jiang/PGGAN-PyTorch 

[3] Brownlee, J. (2019, October 10). How to implement the Frechet Inception Distance (FID) for evaluating Gans. MachineLearningMastery.com. Retrieved March 20, 2023, from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/#:~:text=The%20FID%20score%20is%20then,*sqrt(C_1*C_2)) 

[4] Neural transfer using pytorch. Neural Transfer Using PyTorch - PyTorch Tutorials 2.0.0+cu117 documentation. (n.d.). Retrieved March 26, 2023, from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
