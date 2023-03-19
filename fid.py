import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.transform import resize
import cv2
import os

image_shape = (512, 512, 3)
model = InceptionV3(include_top=False, pooling='avg', input_shape=image_shape)


def calculate_fid(model, image1, image2):
    """

    :param model:
    :param image1:
    :param image2:
    :return:
    """
    images1 = preprocess_input(image1)
    images2 = preprocess_input(image2)
    images1 = np.tile(images1, (2, 1, 1, 1))
    images2 = np.tile(images2, (2, 1, 1, 1))
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = np.real(sqrtm(sigma1.dot(sigma2)))
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


def fid_eval(model, gen_image, data_images):
    fid_scores = []
    for data_image in data_images:
        fid = calculate_fid(model, gen_image, data_image)
        print('FID: %.3f' % fid)
        fid_scores.append(fid)

    return min(fid_scores)


def read_image(path):
    image = cv2.imread(path)
    image = resize(image, image_shape, 0)
    return image


def data_loader(dataset_path):
    images = []
    for image_name in os.listdir(dataset_path):
        if image_name.endswith('.jpg'):
            image = read_image(os.path.join(dataset_path, image_name))
            images.append(image)

    return images


def run(image_path, dataset_path):
    data_images = data_loader(dataset_path)
    test_image = read_image(image_path)
    fid_score = fid_eval(model, test_image, data_images)
    print('FID: %.3f' % fid_score)


if __name__ == '__main__':
    image_path = 'image_split/distort.png'
    dataset_path = 'fid_eval_dataset/portrait'
    run(image_path, dataset_path)
