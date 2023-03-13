import cv2
import os

image_size = 512
grid_size = 3

def split_image(path, image, image_size, grid_size, padding_size=1):
    for i in range(grid_size):
        for j in range(grid_size):
            x = j * (image_size + padding_size)
            y = i * (image_size + padding_size)
            piece = image[y:y + image_size, x:x + image_size]
            save_name = os.path.join(path, 'piece_{}_{}.jpg'.format(i, j))
            cv2.imwrite(save_name, piece)


def main(path, image_name):
    image = cv2.imread(os.path.join(path, image_name))
    save_path = os.path.join(path, image_name.split('.')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    split_image(save_path, image, image_size, grid_size)


if __name__ == '__main__':
    path = 'image_split'
    image_name = 'portrait_good_1.jpg'
    main(path, image_name)
