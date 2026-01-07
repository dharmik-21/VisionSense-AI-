# Import YOLO model from Ultralytics library
# YOLO (You Only Look Once) is a real-time object detection algorithm
from ultralytics import YOLO


# Load pretrained YOLOv8 nano model
# 'yolov8n.pt' is a lightweight model suitable for fast training and inference
model = YOLO("yolov8n.pt")


# Train the YOLOv8 model
model.train(

    data="E:\project\datasets\detection\dataset.yaml",
    # Path to dataset configuration file
    # This YAML file defines:
    # - Training and validation image paths
    # - Number of classes
    # - Class names

    epochs=10,
    # Number of full passes through the training dataset
    # More epochs generally improve accuracy but increase training time

    imgsz=640,
    # Input image size used for training
    # Larger size improves detection of small objects but uses more memory

    batch=16,
    # Number of images processed together in one training iteration
    # Higher batch size = faster training (if enough memory is available)

    project="E:\project\models",
    # Directory where trained models and logs will be saved

    name="detector"
    # Subfolder name inside project directory
    # Final path example:
    # E:\project\models\detector\
)
