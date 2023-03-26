"""
    This is the additional works for the project "Art Investigate with Generative Adversarial Network" from Junting Li.
    University College London, 2023.

    This is the user interface for the style transfer.
"""

import sys
import math
from PyQt5.Qt import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget
import copy
from functions import read_image, centroid_crop, resize_image, convert_to_pixmap
from style_transfer import style_transfer
import torch
import cv2


def calibrate(pos, factor):
    return math.floor(pos[0] * factor[0]), math.floor(pos[1] * factor[1])


class data_storage:
    style_image = None
    content_image = None


class Image_Display(QWidget):
    def __init__(self, image, calibrate_factor):
        super().__init__()
        self.raw_image = copy.deepcopy(image)
        self.calibrate_factor = calibrate_factor
        self.image_size = calibrate((800, 800), self.calibrate_factor)
        self.display_size = calibrate((800, 900), self.calibrate_factor)
        self.setGeometry(*calibrate((50, 50), self.calibrate_factor), *self.display_size)
        self.setWindowTitle("Display image")
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.setFixedSize(self.size())

        img_display = resize_image(image, self.image_size)
        img_display = convert_to_pixmap(img_display)
        self.label = QLabel(self)
        self.label.setGeometry(*calibrate((25, 25), self.calibrate_factor),
                               *calibrate((750, 750), self.calibrate_factor))
        self.label.setPixmap(img_display)

        # add a button to save the image
        self.save_button = QPushButton(self)
        self.save_button.setGeometry(*calibrate((290, 820), self.calibrate_factor),
                                     *calibrate((200, 50), self.calibrate_factor))
        self.save_button.setText("Save")
        self.save_button.clicked.connect(self.save_image)
        print("Image_Display init finished")

        self.show()

    def save_image(self):
        file_name = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.bmp)")[0]
        if file_name:
            if not file_name.endswith(".jpg") or file_name.endswith(".png"):
                file_name += ".jpg"
            try:
                print("Saving image to {}".format(file_name))
                save_image = cv2.cvtColor(copy.deepcopy(self.raw_image), cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_name, save_image)

            except Exception as e:
                print("Error saving image: {}".format(str(e)))


class style_trans_thread(QThread):
    display_image = pyqtSignal(object)
    change_btn_status = pyqtSignal(object, bool)
    msgbox = pyqtSignal(str, str, str, object)

    def __init__(self, content_image, style_image, btn_object, device, step):
        super(style_trans_thread, self).__init__()
        self.content_image = content_image
        self.style_image = style_image
        self.btn_object = btn_object
        self.device = device
        self.num_steps = step

    def run(self):
        self.change_btn_status.emit(self.btn_object, False)
        try:
            style_trans = style_transfer(self.content_image, self.style_image)
            output_image = style_trans.run(self.num_steps, self.device)
            self.display_image.emit(output_image)

            del style_trans
            torch.cuda.empty_cache()

        except Exception as ex:
            self.msgbox.emit("Error", "The style transfer failed", str(ex), QMessageBox.Critical)

        finally:
            self.change_btn_status.emit(self.btn_object, True)


class APP(QMainWindow):
    def __init__(self, calibrate_factor):
        super().__init__()

        if not torch.cuda.is_available():
            self.msgbox("Error", "No GPU detected", "Please check your GPU", QMessageBox.Critical)
            sys.exit(0)

        self.temp = data_storage()
        self.calibrate_factor = calibrate_factor
        self.img_display_size = calibrate((600, 600), self.calibrate_factor)
        print("Grid image size: ", self.img_display_size)

        self.setGeometry(*calibrate((50, 50), self.calibrate_factor), *calibrate((1350, 900), self.calibrate_factor))
        self.setWindowTitle("Image Style Transfer")
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.setFixedSize(self.size())
        self.GPUs = torch.cuda.device_count()
        self.toolbar_offset = 300
        self.setup_labels()
        self.setup_ui()

        self.setStyleSheet("background-color: rgb(240, 240, 255);")
        self.show()

    def setup_labels(self):
        self.content_label = QLabel(self)
        self.content_label.setText("Content Image:")
        self.content_label.setGeometry(*calibrate((20, 30), self.calibrate_factor),
                                       *calibrate((200, 30), self.calibrate_factor))
        self.content_label.setFont(QFont("Times", 14, QFont.Bold))

        self.style_label = QLabel(self)
        self.style_label.setText("Style Image:")
        self.style_label.setGeometry(*calibrate((700, 30), self.calibrate_factor),
                                     *calibrate((200, 30), self.calibrate_factor))
        self.style_label.setFont(QFont("Times", 14, QFont.Bold))

        self.gpu_label = QLabel(self)
        self.gpu_label.setText("GPU:")
        self.gpu_label.setGeometry(*calibrate((0 + self.toolbar_offset, 760), self.calibrate_factor),
                                   *calibrate((100, 50), self.calibrate_factor))
        self.gpu_label.setFont(QFont("Arial", 10, QFont.Normal))

        self.step_label = QLabel(self)
        self.step_label.setText("Optimization Steps:")
        self.step_label.setGeometry(*calibrate((300 + self.toolbar_offset, 760), self.calibrate_factor),
                                    *calibrate((200, 50), self.calibrate_factor))
        self.step_label.setFont(QFont("Arial", 10, QFont.Normal))

        self.step_value_label = QLabel(self)
        self.step_value_label.setGeometry(*calibrate((660 + self.toolbar_offset, 760), self.calibrate_factor),
                                          *calibrate((50, 50), self.calibrate_factor))
        self.step_value_label.setFont(QFont("Arial", 10, QFont.Normal))

    def setup_ui(self):
        # Content image part
        self.content_img = QLabel(self)
        self.content_img.setGeometry(*calibrate((35, 70), self.calibrate_factor), *self.img_display_size)

        self.btn_content = QPushButton(self)
        self.btn_content.setText("Select Content Image")
        self.btn_content.setGeometry(*calibrate((225, 690), self.calibrate_factor),
                                     *calibrate((180, 40), self.calibrate_factor))
        self.btn_content.clicked.connect(self.btn_content_clicked)

        # Style image part
        self.style_img = QLabel(self)
        self.style_img.setGeometry(*calibrate((715, 70), self.calibrate_factor), *self.img_display_size)

        self.btn_style = QPushButton(self)
        self.btn_style.setText("Select Style Image")
        self.btn_style.setGeometry(*calibrate((925, 690), self.calibrate_factor),
                                   *calibrate((180, 40), self.calibrate_factor))
        self.btn_style.clicked.connect(self.btn_style_clicked)

        # paint rectangle side
        self.paintEvent(QPaintEvent)

        # Generate button
        self.btn_generate = QPushButton(self)
        self.btn_generate.setText("Generate")
        self.btn_generate.setGeometry(*calibrate((540, 840), self.calibrate_factor),
                                      *calibrate((280, 40), self.calibrate_factor))
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self.btn_generate_clicked)

        # select GPU
        self.gpu_device = QComboBox(self)
        self.gpu_device.setGeometry(*calibrate((50 + self.toolbar_offset, 760), self.calibrate_factor),
                                    *calibrate((200, 50), self.calibrate_factor))
        self.gpu_device.addItems(["cuda:{}".format(str(i)) for i in range(self.GPUs)])

        # adjust step slider
        self.step_slider = QSlider(Qt.Horizontal, self)
        self.step_slider.setGeometry(*calibrate((450 + self.toolbar_offset, 760), self.calibrate_factor),
                                     *calibrate((200, 50), self.calibrate_factor))
        self.step_slider.setMinimum(50)
        self.step_slider.setMaximum(400)
        self.step_slider.setValue(200)
        self.step_value_label.setText("200")
        self.step_slider.valueChanged.connect(self.step_slider_changed)

    def btn_content_clicked(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp)")
        if img_path != "":
            try:
                img = read_image(img_path)
                img = centroid_crop(img)
                self.temp.content_image = copy.deepcopy(img)
                img_display = resize_image(img, self.img_display_size)
                img_display = convert_to_pixmap(img_display)
                self.content_img.setPixmap(img_display)
                self.update_button_state()

            except Exception as ex:
                self.msgbox("Error", "There is an error occurred", str(ex), QMessageBox.Critical)

    def btn_style_clicked(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp)")
        if img_path != "":
            try:
                img = read_image(img_path)
                img = centroid_crop(img)
                self.temp.style_image = img
                img_display = resize_image(img, self.img_display_size)
                img_display = convert_to_pixmap(img_display)
                self.style_img.setPixmap(img_display)
                self.update_button_state()

            except Exception as ex:
                self.msgbox("Error", "There is an error occurred", str(ex), QMessageBox.Critical)

    def update_button_state(self):
        if self.temp.content_image is not None and self.temp.style_image is not None:
            self.btn_generate.setEnabled(True)
        else:
            self.btn_generate.setEnabled(False)

    def change_btn_state(self, btn, state):
        btn.setEnabled(state)

    def btn_generate_clicked(self):
        content_img = copy.deepcopy(self.temp.content_image)
        style_img = resize_image(copy.deepcopy(self.temp.style_image), content_img.shape[:2])
        device = torch.device(self.gpu_device.currentText())
        num_step = self.step_slider.value()

        self.thread = style_trans_thread(content_img, style_img, self.btn_generate, device, num_step)
        self.thread.display_image.connect(self.display)
        self.thread.change_btn_status.connect(self.change_btn_state)
        self.thread.msgbox.connect(self.msgbox)
        self.thread.start()
        print("Thread started")

    def display(self, image):
        self.output_window = Image_Display(image, self.calibrate_factor)
        self.output_window.show()

    def msgbox(self, title, text, detailed_text="", icon=QMessageBox.Information):
        self.msg = QMessageBox()
        self.msg.setIcon(icon)
        self.msg.setText(text)
        self.msg.setWindowTitle(title)
        self.msg.setDetailedText(detailed_text)
        self.msg.setStandardButtons(QMessageBox.Ok)
        self.msg.exec_()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 5, Qt.DotLine))

        start_point = calibrate((35, 70), self.calibrate_factor)
        painter.drawRect(*start_point, *self.img_display_size)

        start_point = calibrate((715, 70), self.calibrate_factor)
        painter.drawRect(*start_point, *self.img_display_size)

    def step_slider_changed(self):
        self.step_value_label.setText(str(self.step_slider.value()))


def screen_resize_factor(width, height):
    base = (1707, 1067)
    return width / base[0], height / base[1]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    size = screen.size()
    print('Screen Size: %d x %d' % (size.width(), size.height()))
    factor = screen_resize_factor(size.width(), size.height())
    ex = APP(factor)
    sys.exit(app.exec_())
