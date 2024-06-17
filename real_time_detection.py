import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torchvision.models as models
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QTextBrowser, QGridLayout, QGroupBox, QStyleFactory
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
import sys
from PyQt5.QtWidgets import QHBoxLayout, QStyle

class PlantDiseaseDetection(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        # Set device to use GPU if available, otherwise use CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the trained model
        self.model_path = os.path.join("models", "plant_disease_model.pth")
        self.model = models.resnet18()  # Instantiate the ResNet-18 model
        self.num_classes = 7  # Update the number of classes to match your dataset
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)  # Replace the final layer
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.model = self.model.to(self.device)

        # Define data transformations (resizing, normalization)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define class labels
        self.class_labels = ['plant', 'bercak_daun', 'defisiensi_kalsium', 'hangus_daun', 'hawar_daun', 'mosaik_vena_kuning', 'virus_kuning_keriting']

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)

        self.detection_running = False
        self.timer = QTimer()

        self.show()

    def initUI(self):
        self.setWindowTitle('Plant Disease Detection')
        self.setGeometry(100, 100, 800, 600)

        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        self.group_box = QGroupBox('Plant Disease Detection')
        self.group_box.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 10px; padding: 10px;")
        self.grid_layout.addWidget(self.group_box, 0, 0)

        self.group_box_layout = QVBoxLayout()
        self.group_box.setLayout(self.group_box_layout)

        self.label = QLabel()
        self.label.setStyleSheet("background-color: #fff; border: 1px solid #ccc; border-radius: 10px; padding: 10px;")
        self.group_box_layout.addWidget(self.label)

        font = QFont("Arial", 24, QFont.Bold)
        self.detection_label = QLabel()
        self.detection_label.setFont(font)
        self.detection_label.setAlignment(Qt.AlignCenter)
        self.detection_label.setStyleSheet("background-color: #fff; border: 1px solid #ccc; border-radius: 10px; padding: 10px;")
        self.group_box_layout.addWidget(self.detection_label)

        self.text_browser = QTextBrowser()
        self.text_browser.setStyleSheet("background-color: #fff; border: 1px solid #ccc; border-radius: 10px; padding: 10px;")
        self.group_box_layout.addWidget(self.text_browser)

        self.button_layout = QHBoxLayout()
        self.group_box_layout.addLayout(self.button_layout)

        self.button_start = QPushButton('Start Detection')
        self.button_start.setStyleSheet("background-color: #4CAF50; color: #fff; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;")
        self.button_start.clicked.connect(self.startDetection)
        self.button_layout.addWidget(self.button_start)

        self.button_stop = QPushButton('Stop Detection')
        self.button_stop.setStyleSheet("background-color: #f44336; color: #fff; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;")
        self.button_stop.clicked.connect(self.stopDetection)
        self.button_layout.addWidget(self.button_stop)

        self.button_stop.setEnabled(False)

    def startDetection(self):
        self.detection_running = True
        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.timer.timeout.connect(self.detectDisease)
        self.timer.start(30)  # 30 milliseconds

    def stopDetection(self):
        self.detection_running = False
        self.button_start.setEnabled(True)
        self.button_stop.setEnabled(False)
        self.timer.stop()

    def detectDisease(self):
        if self.detection_running:
            ret, frame = self.cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            image = self.transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = self.class_labels[predicted.item()]

            if predicted_label != 'plant':  # Only show label if disease is detected
                cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                self.label.setPixmap(QPixmap.fromImage(qImg))

                self.detection_label.setText(predicted_label)
                self.text_browser.setText(predicted_label)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.detection_running = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PlantDiseaseDetection()
    sys.exit(app.exec_())