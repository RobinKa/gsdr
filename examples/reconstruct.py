import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from gsdr import GSDRStack
import numpy as np
from sklearn.datasets import *
import PIL
import PIL.ImageQt
import random
import math

def get_olivetti_faces():
    faces = fetch_olivetti_faces()
    
    faces.data = faces.data.astype(np.float32)
    faces.target = faces.target.astype(np.int32)

    return faces.data, faces.target, (64, 64)

def get_lfw():
    lfw = fetch_lfw_people(resize=1)
    
    lfw.data = lfw.data.astype(np.float32) / 255.0
    lfw.target = lfw.target.astype(np.int32)

    return lfw.data, lfw.target, (125, 94)

def get_mnist():
    mnist = fetch_mldata('MNIST original')
    
    # Normalize between 0 and 1
    mnist.data = mnist.data.astype(np.float32) / 255.0
    mnist.target = mnist.target.astype(np.int32)

    return mnist.data, mnist.target, (28, 28)

def pil_to_q_image(pil_image):
    image_q = PIL.ImageQt.ImageQt(pil_image)
    q_image = QImage(image_q)
    return q_image

class SDRWidget(QWidget):
    def __init__(self, hidden_count, reconstruct_func):
        super().__init__()

        self.layout = QGridLayout(self)

        self.reconstruct_func = reconstruct_func
        self.labels = []

        self.hidden_count = hidden_count

        wh = math.ceil(math.sqrt(hidden_count))

        for i in range(hidden_count):
            label = QPushButton(self)
            label.sdr_index = i
            
            label.clicked.connect(self.toggle_sdr)
            self.labels.append(label)

            self.layout.addWidget(label, i % wh, i // wh)

        self.set_sdr(np.zeros(hidden_count))
        self.setLayout(self.layout)

    def set_sdr(self, sdr):
        self.sdr = np.array(sdr)

        for i in range(self.hidden_count):
            color = "blue" if sdr[i] == 1 else "white"
            self.labels[i].setStyleSheet("QPushButton { background-color: %s; }" % color);

    def toggle_sdr(self):
        i = self.sender().sdr_index
        self.sdr[i] = 1 - self.sdr[i]
        color = "blue" if self.sdr[i] == 1 else "white"
        self.labels[i].setStyleSheet("QPushButton { background-color: %s; }" % color);
        self.reconstruct_func()

class ReconstructWindow(QWidget):
    def __init__(self, gsdr, train_data, test_data, image_shape, train_count=1000):
        super().__init__()

        self.hidden_count = gsdr._layers_reversed[0].hidden_count

        self.image_shape = image_shape

        self.train_count = train_count

        self.train_data = train_data
        self.test_data = test_data

        self.gsdr = gsdr

        self._init_ui()

    def _init_ui(self):
        self.resize(600, 260)
        self.setWindowTitle('Reconstruct')

        self.layout = QGridLayout(self)

        self.original_pixmap = QPixmap()
        self.original_image = QLabel(self)
        self.original_image.setPixmap(self.original_pixmap)
        #self.original_image.setScaledContents(True)
        self.layout.addWidget(self.original_image, 0, 0)

        self.reconstructed_pixmap = QPixmap()
        self.reconstructed_image = QLabel(self)
        self.reconstructed_image.setPixmap(self.reconstructed_pixmap)
        #self.reconstructed_image.setScaledContents(True)
        self.layout.addWidget(self.reconstructed_image, 0, 1)

        self.sdr_widget = SDRWidget(self.hidden_count, self.reconstruct_sdr)
        self.layout.addWidget(self.sdr_widget, 0, 2)

        self.reconstruct_button = QPushButton("Reconstruct", self)
        self.reconstruct_button.clicked.connect(self.visualize_image)
        self.layout.addWidget(self.reconstruct_button, 1, 0)

        self.train_button = QPushButton("Train", self)
        self.train_button.clicked.connect(self.train)
        self.layout.addWidget(self.train_button, 1, 1)

        self.setLayout(self.layout) 

    def train(self):
        for i in range(self.train_count):
            d = self.train_data[np.random.randint(0, self.train_data.shape[0])]
            self.gsdr.train(d, 0.003)

    def visualize_image(self):
        original_image = random.choice(self.test_data)

        reconstructed_image = self.gsdr.get_reconstructed(original_image)

        original_image = (original_image.reshape(self.image_shape) * 255).astype(np.int8)
        reconstructed_image = (np.clip(reconstructed_image.reshape(self.image_shape), 0, 1) * 255).astype(np.int8)

        self.original_pixmap = QPixmap(pil_to_q_image(PIL.Image.fromarray(original_image, "L")))
        self.original_image.setPixmap(self.original_pixmap)

        self.reconstructed_pixmap = QPixmap(pil_to_q_image(PIL.Image.fromarray(reconstructed_image, "L")))
        self.reconstructed_image.setPixmap(self.reconstructed_pixmap)

        self.sdr_widget.set_sdr(self.gsdr._layers_reversed[0].state)

    def reconstruct_sdr(self):
        reconstructed_image = self.gsdr.generate(np.array(self.sdr_widget.sdr))
        reconstructed_image = (np.clip(reconstructed_image.reshape(self.image_shape), 0, 1) * 255).astype(np.int8)

        self.reconstructed_pixmap = QPixmap(pil_to_q_image(PIL.Image.fromarray(reconstructed_image, "L")))
        self.reconstructed_image.setPixmap(self.reconstructed_pixmap)

def main():
    app = QApplication(sys.argv)

    data, targets, image_shape = get_olivetti_faces()

    data = data[np.random.permutation(len(data))]

    train_data = []
    test_data = []

    for d, t in zip(data, targets):
        if t < 36:
            train_data.append(d)
        else:
            test_data.append(d)

    print("Train data count:", len(train_data))
    print("Test data count:", len(test_data))

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # Construct the network
    gsdr = GSDRStack()
    gsdr.add(input_count=train_data.shape[1], hidden_count=256, sparsity=0.20)
    gsdr.add(hidden_count=256, sparsity=0.15)
    gsdr.add(hidden_count=256, sparsity=0.10)
    gsdr.add(hidden_count=256, sparsity=0.05)

    wnd = ReconstructWindow(gsdr, train_data, test_data, image_shape, train_count=1000)
    wnd.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()