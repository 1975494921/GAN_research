import torch
from torchvision import transforms
import os
from PIL import Image

target_img_size = 256
image_dir = "D:\dataset\impressionist_landscapes_resized_1024"
save_dir = 'D:\dataset\img_size_256\impressive_landscape_256'
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)

# for path, dir_list, file_list in os.walk(image_dir):
#     for file_name in tqdm.tqdm(file_list):
#         file_path = os.path.join(path, file_name)
#         if file_path.endswith(".jpg") or file_path.endswith(".png"):
#             try:
#                 img = cv2.imread(file_path)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 min_size = min(img.shape[0], img.shape[1])
#                 if min_size < target_img_size:
#                     print("Discard {}: {}".format(file_name, img.shape))
#                 else:
#                     save_path = os.path.join(save_dir, file_name)
#                     transform = transforms.Compose([
#                         transforms.ToTensor(),  # Converts to PyTorch Tensor
#                         transforms.CenterCrop(target_img_size),
#                         transforms.ToPILImage()  # converts the tensor to PIL image
#                     ])
#
#                     img = transform(img)
#                     img.save(save_path)
#                     # print("Save {}".format(save_path))
#
#             except Exception as e:
#                 print("Discard {} Error:{}".format(file_name, str(e)))


def delete_task():
    for path, dir_list, file_list in os.walk(image_dir):
        for file_name in file_list:
            if file_name.endswith(".csv"):
                file_path = os.path.join(path, file_name)
                print("Delete {}".format(file_name))
                os.remove(file_path)
