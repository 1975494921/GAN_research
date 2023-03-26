import cv2
from PyQt5 import QtGui


def read_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    except Exception as ex:
        raise ex


def centroid_crop(img):
    corp_size = min(img.shape[0], img.shape[1])
    x = (img.shape[0] - corp_size) // 2
    y = (img.shape[1] - corp_size) // 2
    return img[x:x + corp_size, y:y + corp_size]


def resize_image(img, size):
    return cv2.resize(img, size)


def convert_to_pixmap(img):
    height, width, channel = img.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qImg)
    return pixmap
